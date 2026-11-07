#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run compute shader A/B cases: equal outputs and GPU time for a baseline and candidates.

    shader_ab.py run CASE.json... [--set NAME=VALUE] [--subgroup N] [--cases N] [--timing N]
                                   [--samples N] [--device N] [--validate] [--cold-compile]
                                   [--json OUT.json] [--runner PATH]
    shader_ab.py baseline [--out DIR] [--build-dir DIR] [--no-build] [--jobs N]
    shader_ab.py check CASE.json... [same options as run and baseline]
    shader_ab.py fixtures --out DIR [--build-dir DIR] [--no-build] [--filter TAG]
    shader_ab.py reflect MODULE.spv [--runner PATH]
    shader_ab.py build [--runner PATH]

`baseline` rebuilds the unit tests, emits every [fixture] test's SPIR-V into the baseline
directory and records the commit it came from. `check` does the same into the candidate
directory and runs the cases with ${BEFORE} and ${AFTER} pointing at the two directories.

A case file holds one object or a list of them. Module specs are a path to a .spv or
.comp file, or {"spv": PATH} / {"glsl": PATH, "defines": {...}, "args": [...], "name": ...}.
Paths are relative to the case file and expand ${NAME} from --set or the environment.
See tools/nx2/shader-ab.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = REPO / "tools" / "shader-ab" / "shader_ab.cpp"
DEFAULT_RUNNER = REPO / "build" / "shader-ab" / "shader-ab"
DEFAULT_BUILD_DIR = REPO / "build" / "dev"
BASELINE_DIR = REPO / "build" / "shader-ab" / "baseline"
CANDIDATE_DIR = REPO / "build" / "shader-ab" / "candidate"
BUFFER_KEYS = ("set", "binding", "size", "fill", "file", "compare", "tolerance", "begin", "end")
COLD_COMPILE_ENV = {"RADV_DEBUG": "nocache", "MESA_SHADER_CACHE_DISABLE": "true",
                    "MESA_GLSL_CACHE_DISABLE": "true", "__GL_SHADER_DISK_CACHE": "0"}


class CaseError(ValueError):
    pass


def expand(text, variables):
    """Expand ${NAME} and $NAME; an unset name is an error, not an empty string."""
    def replace(match):
        name = match.group(1) or match.group(2)
        if name not in variables:
            raise CaseError(f"${name} is not set; pass --set {name}=... or export it")
        return variables[name]
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, text)


def resolve_path(text, base_dir, variables):
    path = Path(os.path.expanduser(expand(text, variables)))
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_runner(runner, source=RUNNER_SOURCE):
    """Compile the Vulkan runner when it is missing or older than its source."""
    runner = Path(runner)
    if runner.exists() and runner.stat().st_mtime >= source.stat().st_mtime:
        return runner
    runner.parent.mkdir(parents=True, exist_ok=True)
    command = [os.environ.get("CXX", "c++"), "-std=c++20", "-O2", str(source), "-lvulkan",
               "-o", str(runner)]
    print("Building runner:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)
    return runner


def glsl_command(source, output, defines=None, args=None, stage=None):
    command = [shutil.which("glslangValidator") or "glslangValidator", "-V",
               "--target-env", "vulkan1.3"]
    if stage:
        command += ["-S", stage]
    for name, value in sorted((defines or {}).items()):
        command.append(f"-D{name}" if value is None else f"-D{name}={value}")
    command += list(args or [])
    command += ["-o", str(output), str(source)]
    return command


def compile_glsl(source, defines, args, cache_dir):
    """Compile GLSL to SPIR-V with glslangValidator and validate it when spirv-val exists."""
    identity = hashlib.sha1(json.dumps([str(source), source.stat().st_mtime, defines, args],
                                       sort_keys=True).encode()).hexdigest()[:16]
    output = Path(cache_dir) / f"{source.stem}-{identity}.spv"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = None if source.suffix in (".comp",) else "comp"
        subprocess.run(glsl_command(source, output, defines, args, stage), check=True)
        validator = shutil.which("spirv-val")
        if validator:
            subprocess.run([validator, "--target-env", "vulkan1.3", str(output)], check=True)
    return output


def module_spec(spec, default_name):
    """Normalize a module spec to {"name", "spv"|"glsl", "defines", "args"}."""
    if isinstance(spec, str):
        key = "glsl" if spec.endswith((".comp", ".glsl")) else "spv"
        spec = {key: spec}
    if not isinstance(spec, dict) or ("spv" in spec) == ("glsl" in spec):
        raise CaseError(f"Module spec needs exactly one of spv/glsl: {spec!r}")
    defines = spec.get("defines", {})
    name = spec.get("name") or (
        ",".join(f"{k}={v}" for k, v in sorted(defines.items())) if defines else default_name)
    return {"name": name, "spv": spec.get("spv"), "glsl": spec.get("glsl"),
            "defines": defines, "args": list(spec.get("args", []))}


def resolve_module(spec, base_dir, variables, cache_dir):
    if spec["spv"]:
        path = resolve_path(spec["spv"], base_dir, variables)
        if not path.exists():
            raise CaseError(f"Module not found: {path}")
        return path
    return compile_glsl(resolve_path(spec["glsl"], base_dir, variables), spec["defines"],
                        spec["args"], cache_dir)


def words_image(words):
    """Build the bytes for a `words` entry: a list at offset 0, or {byte offset: list}."""
    table = words if isinstance(words, dict) else {0: words}
    image = bytearray()
    for offset, values in table.items():
        offset = int(offset, 0) if isinstance(offset, str) else int(offset)
        if offset % 4:
            raise CaseError(f"words offset {offset} is not a multiple of 4")
        end = offset + 4 * len(values)
        if end > len(image):
            image.extend(bytes(end - len(image)))
        for index, value in enumerate(values):
            value = int(value, 0) if isinstance(value, str) else int(value)
            image[offset + 4 * index:offset + 4 * index + 4] = (value & 0xffffffff).to_bytes(4, "little")
    return bytes(image)


def words_file(words, cache_dir):
    image = words_image(words)
    path = Path(cache_dir) / f"words-{hashlib.sha1(image).hexdigest()[:16]}.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image)
    return path


def buffer_argument(buffer, base_dir, variables, cache_dir=None):
    unknown = set(buffer) - set(BUFFER_KEYS) - {"words"}
    if unknown:
        raise CaseError(f"Unknown buffer keys: {sorted(unknown)}")
    if "words" in buffer and ("file" in buffer or "fill" in buffer):
        raise CaseError("words excludes fill and file")
    parts = []
    for key in BUFFER_KEYS:
        if key not in buffer:
            continue
        value = buffer[key]
        if key == "file":
            value = resolve_path(value, base_dir, variables)
        parts.append(f"{key}={value}")
    if "words" in buffer:
        parts.append(f"file={words_file(buffer['words'], cache_dir)}")
    return ",".join(parts)


def runner_arguments(case, baseline, candidate, subgroup, base_dir, variables, overrides,
                     cache_dir=None):
    args = ["--baseline", str(baseline), "--candidate", str(candidate)]
    dispatch = case.get("dispatch", [1, 1, 1])
    if len(dispatch) != 3:
        raise CaseError("dispatch needs three workgroup counts")
    args += ["--dispatch", ",".join(str(int(n)) for n in dispatch)]
    if subgroup:
        args += ["--subgroup", str(subgroup)]
    if case.get("entry"):
        args += ["--entry", case["entry"]]
    for buffer in case.get("buffers", []):
        args += ["--buffer", buffer_argument(buffer, base_dir, variables, cache_dir)]
    push = case.get("push")
    if isinstance(push, dict):
        args += ["--push", f"file={resolve_path(push['file'], base_dir, variables)}"]
    elif push:
        args += ["--push", f"hex={push}"]
    timing = case.get("timing", 0)
    if isinstance(timing, dict):
        samples = timing.get("samples")
        timing = timing.get("repeats", 0)
    else:
        samples = None
    cases = overrides.get("cases") or case.get("cases")
    if cases:
        args += ["--cases", str(cases)]
    timing = overrides.get("timing") if overrides.get("timing") is not None else timing
    if timing:
        args += ["--timing", str(timing)]
    samples = overrides.get("samples") or samples
    if samples:
        args += ["--samples", str(samples)]
    if case.get("show") is not None:
        args += ["--show", str(case["show"])]
    if overrides.get("device"):
        args += ["--device", str(overrides["device"])]
    if overrides.get("validate"):
        args.append("--validate")
    return args


def load_cases(path):
    with open(path) as file:
        data = json.load(file)
    cases = data if isinstance(data, list) else [data]
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or "baseline" not in case:
            raise CaseError(f"{path}: case {index} needs a baseline")
        if ("candidate" in case) == ("candidates" in case):
            raise CaseError(f"{path}: case {index} needs candidate or candidates")
        case.setdefault("name", f"{Path(path).stem}#{index}")
    return cases


def candidates_of(case):
    specs = case["candidates"] if "candidates" in case else [case["candidate"]]
    return [module_spec(spec, f"candidate{i}" if len(specs) > 1 else "candidate")
            for i, spec in enumerate(specs)]


def subgroups_of(case, override):
    if override is not None:
        return [override]
    value = case.get("subgroup", 0)
    return list(value) if isinstance(value, list) else [value]


def parse_report(text):
    return json.loads(text)


def outcome(report, expect):
    failed = report["mismatches"] > 0 or report.get("validation_errors", 0) > 0
    if expect == "fail":
        return ("DETECTED", True) if failed else ("UNDETECTED", False)
    return ("FAIL", False) if failed else ("PASS", True)


def format_rows(rows):
    header = ["case", "candidate", "sg", "result", "compared", "max dist", "base ns",
              "cand ns", "ratio", "pipe ms"]
    table = [header]
    for row in rows:
        report = row.get("report")
        if report is None:
            table.append([row["case"], row["candidate"], str(row["subgroup"]), "ERROR",
                          "", "", "", "", "", ""])
            continue
        timing = report.get("timing")
        table.append([
            row["case"], row["candidate"], str(report["subgroup"]), row["result"],
            str(report["compared"]), f"{report['max_distance']:g}",
            f"{timing['baseline']['median']:.1f}" if timing else "",
            f"{timing['candidate']['median']:.1f}" if timing else "",
            f"{timing['ratio']:.3f}" if timing else "",
            f"{report['pipeline_ms'][0]:.1f}/{report['pipeline_ms'][1]:.1f}"])
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip()
                     for line in table)


def git_state():
    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(REPO)] + list(command), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": git("rev-parse", "HEAD"), "describe": git("describe", "--always", "--dirty"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def emit_cache_compute_fixtures(cache_path, out_dir, build_dir, build, jobs):
    """Compile compute environments from a serialized shader cache into cache-<hash>.spv."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shader_sweep as sweep
    cache_path = Path(os.path.expanduser(cache_path))
    if not cache_path.is_file():
        raise CaseError(f"Shader cache not found: {cache_path}")
    if build:
        command = ["cmake", "--build", str(build_dir), "--target", "shader-sweep",
                   "--parallel", str(jobs)]
        print("$", " ".join(command), file=sys.stderr)
        subprocess.run(command, check=True)
    compiler = Path(build_dir) / "bin" / "shader-sweep"
    if not compiler.is_file():
        raise CaseError(f"shader-sweep binary not found: {compiler}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for env in sweep.parse_cache(cache_path.read_bytes()):
            if env["stage"] != "compute" or len(env["key"]) < 8:
                continue
            shader_hash = f"{int.from_bytes(env['key'][:8], 'little'):016x}"
            env_file = tmp_dir / f"{shader_hash}.env"
            env_file.write_bytes(env["bytes"])
            spv_out = Path(out_dir) / f"cache-{shader_hash}.spv"
            subprocess.run([str(compiler), "--environment", str(env_file),
                            "--cache-version", str(env["version"]),
                            "--radv-profile", "--output", str(spv_out)],
                           check=True, stdout=subprocess.DEVNULL)


def emit_fixtures(out_dir, build_dir, build, jobs, test_filter, label, cache_path=None):
    """Rebuild the unit tests, run the fixture-emitting ones and tag the output directory."""
    out_dir = Path(out_dir)
    build_dir = Path(build_dir)
    if build:
        command = ["cmake", "--build", str(build_dir), "--target", "tests", "--parallel", str(jobs)]
        print("$", " ".join(command), file=sys.stderr)
        subprocess.run(command, check=True)
    tests = build_dir / "bin" / "tests"
    if not tests.exists():
        raise CaseError(f"Test binary not found: {tests}")
    if out_dir.exists():
        for stale in out_dir.glob("*.spv"):
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EDEN_SHADER_FIXTURES=str(out_dir))
    print("$", f"EDEN_SHADER_FIXTURES={out_dir} {tests} '{test_filter}'", file=sys.stderr)
    subprocess.run([str(tests), test_filter], env=env, check=True)
    if cache_path:
        emit_cache_compute_fixtures(cache_path, out_dir, build_dir, build, jobs)
    record = dict(git_state(), label=label, tests=str(tests), filter=test_filter,
                  time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  fixtures=sorted(p.name for p in out_dir.glob("*.spv")))
    if not record["fixtures"]:
        raise CaseError(f"No fixtures were written to {out_dir}; check the test filter")
    (out_dir / "fixtures.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def describe_fixtures(directory):
    try:
        record = json.loads((Path(directory) / "fixtures.json").read_text())
    except (OSError, ValueError):
        return None
    state = record.get("describe", "?") + (" (dirty)" if record.get("dirty") else "")
    return f"{record.get('label', '?')}: {state}, {len(record.get('fixtures', []))} fixtures, {record.get('time', '?')}"


def default_variables(variables):
    """Point ${BEFORE} and ${AFTER} at the baseline and candidate directories when unset."""
    for name, directory in (("BEFORE", BASELINE_DIR), ("AFTER", CANDIDATE_DIR)):
        if name not in variables and (directory / "fixtures.json").exists():
            variables[name] = str(directory)
    return variables


def run_cases(args):
    variables = dict(os.environ)
    for item in args.set or []:
        name, _, value = item.partition("=")
        variables[name] = value
    default_variables(variables)
    for name in ("BEFORE", "AFTER"):
        if name in variables and (info := describe_fixtures(variables[name])):
            print(f"{name}={variables[name]}  [{info}]", file=sys.stderr)
    runner = build_runner(args.runner)
    env = dict(os.environ)
    if args.cold_compile:
        env.update(COLD_COMPILE_ENV)
    overrides = {"cases": args.cases, "timing": args.timing, "samples": args.samples,
                 "device": args.device, "validate": args.validate}
    cache_dir = Path(args.runner).parent / "glsl"
    rows = []
    all_ok = True
    for case_path in args.cases_files:
        base_dir = Path(case_path).resolve().parent
        for case in load_cases(case_path):
            baseline = resolve_module(module_spec(case["baseline"], "baseline"), base_dir,
                                      variables, cache_dir)
            for candidate in candidates_of(case):
                candidate_path = resolve_module(candidate, base_dir, variables, cache_dir)
                for subgroup in subgroups_of(case, args.subgroup):
                    row = {"case": case["name"], "candidate": candidate["name"],
                           "subgroup": subgroup, "expect": case.get("expect", "pass")}
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                        report_path = handle.name
                    command = [str(runner)] + runner_arguments(
                        case, baseline, candidate_path, subgroup, base_dir, variables,
                        overrides, cache_dir) + ["--json", report_path]
                    print("$", " ".join(command), file=sys.stderr)
                    proc = subprocess.run(command, env=env, capture_output=True, text=True)
                    sys.stderr.write(proc.stderr)
                    if args.verbose:
                        sys.stdout.write(proc.stdout)
                    try:
                        with open(report_path) as file:
                            row["report"] = parse_report(file.read())
                    except (OSError, ValueError):
                        row["report"] = None
                        row["error"] = proc.stderr.strip().splitlines()[-1:] or [
                            f"runner exit {proc.returncode}"]
                        all_ok = False
                    finally:
                        os.unlink(report_path)
                    if row["report"] is not None:
                        row["result"], ok = outcome(row["report"], row["expect"])
                        all_ok &= ok
                    rows.append(row)
    print(format_rows(rows))
    for row in rows:
        if row.get("error"):
            print(f"{row['case']} / {row['candidate']}: {row['error'][0]}")
    if args.json:
        with open(args.json, "w") as file:
            json.dump(rows, file, indent=1)
    return 0 if all_ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER),
                        help="runner binary, built from tools/shader-ab/shader_ab.cpp when stale")
    commands = parser.add_subparsers(dest="command", required=True)
    fixture_options = argparse.ArgumentParser(add_help=False)
    fixture_options.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR),
                                 help="configured CMake build directory with the tests target")
    fixture_options.add_argument("--no-build", action="store_true",
                                 help="use the existing test binary without rebuilding")
    fixture_options.add_argument("--jobs", type=int,
                                 default=int(os.environ.get("LICHII_BUILD_JOBS", os.cpu_count() or 4)))
    fixture_options.add_argument("--filter", default="[fixture]",
                                 help="Catch2 test filter selecting the fixture-emitting tests")
    fixture_options.add_argument("--cache", default=os.environ.get("EDEN_SHADER_CACHE"),
                                 help="optional serialized shader cache to emit cache-<hash>.spv from")
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument("cases_files", nargs="+", metavar="CASE.json")
    run_options.add_argument("--set", action="append", metavar="NAME=VALUE",
                             help="define ${NAME} for paths in the case files")
    for name, parents, help_text in (
            ("run", [run_options], "run case files"),
            ("check", [run_options, fixture_options],
             "emit candidate fixtures from the current tree and run case files against the baseline")):
        run = commands.add_parser(name, parents=parents, help=help_text)
        add_run_options(run)
    baseline = commands.add_parser("baseline", parents=[fixture_options],
                                   help="emit and tag the baseline fixtures from the current tree")
    baseline.add_argument("--out", default=str(BASELINE_DIR))
    fixtures = commands.add_parser("fixtures", parents=[fixture_options],
                                   help="emit fixtures into a directory of your choice")
    fixtures.add_argument("--out", required=True)
    fixtures.add_argument("--label", default="fixtures")
    reflect = commands.add_parser("reflect", help="print a module's compute interface")
    reflect.add_argument("module")
    commands.add_parser("build", help="build the runner")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build_runner(args.runner))
        return 0
    if args.command == "reflect":
        runner = build_runner(args.runner)
        return subprocess.run([str(runner), "--reflect", "--baseline", args.module]).returncode
    if args.command in ("baseline", "fixtures"):
        label = "baseline" if args.command == "baseline" else args.label
        record = emit_fixtures(args.out, args.build_dir, not args.no_build, args.jobs, args.filter,
                               label, args.cache)
        print(f"{label}: {describe_fixtures(args.out)}")
        print("\n".join("  " + name for name in record["fixtures"]))
        return 0
    if args.command == "check":
        if not (BASELINE_DIR / "fixtures.json").exists() and not any(
                item.startswith("BEFORE=") for item in args.set or []):
            raise CaseError(f"No baseline in {BASELINE_DIR}; run `shader_ab.py baseline` on the "
                            "known-good tree first")
        emit_fixtures(CANDIDATE_DIR, args.build_dir, not args.no_build, args.jobs, args.filter,
                      "candidate", args.cache)
    return run_cases(args)


def add_run_options(run):
    run.add_argument("--subgroup", type=int, help="override the required subgroup size")
    run.add_argument("--cases", type=int, help="override the number of input cases")
    run.add_argument("--timing", type=int, help="override serial dispatches per timing sample")
    run.add_argument("--samples", type=int, help="override timing samples per module")
    run.add_argument("--device", type=int, default=0, help="Vulkan physical device index")
    run.add_argument("--validate", action="store_true", help="enable VK_LAYER_KHRONOS_validation")
    run.add_argument("--cold-compile", action="store_true",
                     help="disable the driver shader disk cache so pipeline times are compiles")
    run.add_argument("--json", help="write every row's report to this file")
    run.add_argument("--verbose", action="store_true", help="print the runner's stdout")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CaseError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
