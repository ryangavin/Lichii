#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare two EDEN_NX2_FRAME_DUMP / capture_frame directories for bitwise output equivalence.

    python3 -I tools/nx2/frame_ab.py BASELINE_DIR CANDIDATE_DIR [--present-only] [--json OUT.json]

Reads manifest.jsonl from both directories, pairs every recorded target/storage/present
resource in execution order, decompresses its .raw.zst payload and checks bitwise equality.
Reports the first diverging pass/resource if any output changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def load_manifest(directory: Path):
    manifest_path = directory / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found in {directory}")
    entries = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("kind") in ("frame", "summary"):
            continue
        entries.append(obj)
    return entries


def read_raw_zst(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Missing payload: {path}")
    if path.suffix != ".zst":
        return path.read_bytes()
    zstd = shutil.which("zstd")
    if zstd:
        return subprocess.check_output([zstd, "-d", "-q", "-c", str(path)])
    import zstandard
    return zstandard.ZstdDecompressor().decompress(path.read_bytes())


def compare_bytes(a: bytes, b: bytes):
    if a == b:
        return {"equal": True, "size": len(a), "changed_words": 0, "first_byte_offset": None}
    min_len = min(len(a), len(b))
    first_offset = next((i for i in range(min_len) if a[i] != b[i]), min_len)
    words = min_len // 4
    changed_words = 0
    max_u32_diff = 0
    if words:
        wa = struct.unpack_from(f"<{words}I", a)
        wb = struct.unpack_from(f"<{words}I", b)
        for x, y in zip(wa, wb):
            if x != y:
                changed_words += 1
                diff = abs(x - y)
                if diff > max_u32_diff:
                    max_u32_diff = diff
    return {
        "equal": False,
        "size_a": len(a),
        "size_b": len(b),
        "changed_words": changed_words,
        "total_words": words,
        "first_byte_offset": first_offset,
        "max_u32_diff": max_u32_diff,
    }


def entry_key(entry):
    return (
        entry.get("kind"),
        entry.get("pass"),
        entry.get("index"),
        entry.get("binding"),
        entry.get("format"),
        entry.get("width"),
        entry.get("height"),
        entry.get("depth"),
        tuple(entry.get("shaders") or ()),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--present-only", action="store_true",
                        help="compare only the final presented image(s)")
    parser.add_argument("--json", type=Path, help="write detailed comparison report to JSON")
    args = parser.parse_args(argv)

    base_entries = load_manifest(args.baseline)
    cand_entries = load_manifest(args.candidate)
    if args.present_only:
        base_entries = [e for e in base_entries if e.get("kind") == "present"]
        cand_entries = [e for e in cand_entries if e.get("kind") == "present"]

    if len(base_entries) != len(cand_entries):
        print(f"WARN: entry count differs: baseline={len(base_entries)} vs candidate={len(cand_entries)}",
              file=sys.stderr)

    count = min(len(base_entries), len(cand_entries))
    rows = []
    first_failure = None
    for i in range(count):
        be = base_entries[i]
        ce = cand_entries[i]
        raw_rel_a = be.get("raw") or be.get("file")
        raw_rel_b = ce.get("raw") or ce.get("file")
        row = {
            "index": i,
            "kind": be.get("kind"),
            "pass": be.get("pass"),
            "shaders": be.get("shaders"),
            "format": be.get("format"),
            "extent": [be.get("width"), be.get("height"), be.get("depth", 1)],
            "key_match": entry_key(be) == entry_key(ce),
        }
        if raw_rel_a and raw_rel_b:
            bytes_a = read_raw_zst(args.baseline / raw_rel_a)
            bytes_b = read_raw_zst(args.candidate / raw_rel_b)
            diff = compare_bytes(bytes_a, bytes_b)
            row.update(diff)
            row["sha256_a"] = hashlib.sha256(bytes_a).hexdigest()[:16]
            row["sha256_b"] = hashlib.sha256(bytes_b).hexdigest()[:16]
        else:
            row["equal"] = be.get("stats") == ce.get("stats")
        if not row["equal"] and first_failure is None:
            first_failure = row
        rows.append(row)

    passed = sum(1 for r in rows if r["equal"])
    print(f"Compared {len(rows)} frame-dump resources: {passed}/{len(rows)} bitwise identical")
    for r in rows:
        status = "PASS" if r["equal"] else "DIFF"
        shaders = ",".join(r.get("shaders") or [])
        ext = "x".join(str(x) for x in r["extent"] if x is not None)
        extra = "" if r["equal"] else f" changed_words={r.get('changed_words')}/{r.get('total_words')} first_byte={r.get('first_byte_offset')}"
        print(f"  [{status}] #{r['index']:03d} kind={r['kind']} pass={r['pass']} {ext} fmt={r['format']} shaders=[{shaders}]{extra}")

    if args.json:
        args.json.write_text(json.dumps({
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "total": len(rows),
            "passed": passed,
            "first_failure": first_failure,
            "rows": rows,
        }, indent=2) + "\n")

    return 0 if (first_failure is None and len(base_entries) == len(cand_entries)) else 1


if __name__ == "__main__":
    sys.exit(main())
