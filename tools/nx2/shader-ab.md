# Shader A/B harness

`tools/nx2/shader_ab.py` compares two compute shader implementations on the local Vulkan
device: it dispatches a baseline and a candidate on identical inputs, checks the storage
buffers they write element by element, and times both with GPU timestamps. The runner is
`tools/shader-ab/shader_ab.cpp`, a single-file Vulkan program the driver builds on first use
into `build/shader-ab/shader-ab` (`CXX` selects the compiler). It needs a Vulkan 1.3 device,
`glslangValidator` for GLSL cases and, optionally, `spirv-val` and the validation layer. It does
not start the emulator, read game files or change settings.

## Bit-exact iteration loop

```sh
python3 -I tools/nx2/shader_ab.py baseline              # once, on the known-good tree
# edit the emitter or a pass, then:
python3 -I tools/nx2/shader_ab.py check tools/nx2/shader-ab/global-memory.json \
    tools/nx2/shader-ab/hmma.json
```

`baseline` rebuilds the `tests` target in `build/dev` (`--build-dir`, `--jobs`, `--no-build`),
runs every test tagged `[fixture]` with `EDEN_SHADER_FIXTURES` set, and stores the emitted
modules in `build/shader-ab/baseline` with a `fixtures.json` naming the commit, whether the
tree was dirty, and the time. `check` does the same into `build/shader-ab/candidate` and runs
the cases with `${BEFORE}` and `${AFTER}` pointing at the two directories; it prints both
tags first so a stale baseline is visible. Re-run `baseline` after merging a change you keep.
`run` uses the same defaults when the directories exist, and `fixtures --out DIR` emits into
any directory for keeping several baselines (`--set BEFORE=DIR` selects one).

A fixture is any test that emits SPIR-V through `TestFixtures::WriteSpirv(name, code)` from
`src/tests/video_core/spirv_fixture.h` and carries the `[fixture]` tag. The software HMMA
fixtures (`software-*.spv`) and the global memory routing fixtures (`global-u32-{1,4,8}.spv`,
`global-u32x{2,4}-4.spv`) exist today. A fixture that reads constant buffer or run-time input
needs a matching entry in a case file: see `global-memory.json`, which fills constant buffer 0
with fake NVN buffer descriptors through `words` and leaves the storage buffers random.

## Other commands

```sh
python3 -I tools/nx2/shader_ab.py run tools/nx2/shader-ab/quad-broadcast.json
python3 -I tools/nx2/shader_ab.py reflect module.spv
python3 -I -m unittest discover -s tools/nx2 -p test_shader_ab.py
```

The first command is the self-check: a shuffle and a quad-broadcast variant of one GLSL
source must match (`PASS`), and a variant that drops a lane must be caught (`DETECTED`,
from `"expect": "fail"`). Run it first on a new machine or driver; an `UNDETECTED` or `ERROR`
row means the setup, not the shader, is wrong. The exit status is nonzero when any row fails.

Options override every case in the run: `--subgroup`, `--cases`, `--timing`, `--samples`,
`--device`, `--validate` (requires `VK_LAYER_KHRONOS_validation`; errors fail the row),
`--cold-compile` (disables the driver's shader disk cache so the pipeline column measures
compiles), `--json` (every row's full report) and `--verbose` (the runner's own output).

## Cases

A case file holds one object or a list. Paths are relative to the file and expand `${NAME}`
from `--set NAME=VALUE` or the environment; an unset name is an error.

```json
{
  "name": "software HMMA K16",
  "baseline": "${BEFORE}/software-k16-inputs.spv",
  "candidates": [{"name": "quad", "glsl": "variant.comp", "defines": {"VARIANT": 1}}],
  "dispatch": [1, 1, 1],
  "subgroup": [32, 64],
  "cases": 256,
  "buffers": [{"set": 0, "binding": 0, "size": 2560, "fill": "random-half", "end": 512}],
  "push": "00000040",
  "timing": {"repeats": 1000, "samples": 10},
  "expect": "pass"
}
```

- `baseline`, `candidate` or `candidates`: a `.spv` path, a `.comp`/`.glsl` path, or
  `{"spv": ...}` / `{"glsl": ..., "defines": {...}, "args": [...], "name": ...}`. GLSL compiles
  with `glslangValidator -V --target-env vulkan1.3` and is validated when `spirv-val` exists.
  One source with `#if VARIANT` blocks is the quickest way to try an alternative.
- `dispatch`: workgroup counts. `subgroup`: a required compute subgroup size, or a list to
  run each; `0` is the driver default. `entry` selects a `GLCompute` entry point.
- `buffers`: one entry per descriptor the modules declare (`reflect` lists them). `size` is
  bytes per array element. `fill` is `zero`, `sentinel` (0xcdcdcdcd), `index`, `random`,
  `random-f32`, `random-half` or `file` (a path); `words` instead gives explicit 32-bit words,
  as a list from byte 0 or `{"<byte offset>": [...]}`, with the rest zero (hex strings
  allowed). `compare` is `none`, `bitwise`, `f32-ulp`,
  `f16-ulp` or `f32-abs` with `tolerance`; `begin`/`end` bound the compared bytes. Storage
  buffers default to sentinel fill and bitwise compare, uniform buffers to zero fill and no
  compare. Inputs are deterministic per case index, so a reported mismatch is reproducible.
- `push`: push constant bytes as hex, or `{"file": path}`. `timing`: serial dispatches per
  sample and samples per module; omit or set `0` to skip timing. `show`: mismatches to print.

## Reading the table

- `result`: `PASS`/`FAIL`, or `DETECTED`/`UNDETECTED` for `"expect": "fail"` cases.
  `compared` counts elements over all cases; `max dist` is the largest ULP or absolute
  difference seen. The runner also warns when the baseline left every compared element equal
  to its input, which usually means the dispatch or compared range is wrong.
- `base ns`/`cand ns`: GPU time per dispatch, the median over samples of `repeats` serial
  dispatches with a barrier between each; the order alternates every sample. `ratio` is
  candidate over baseline. `pipe ms` is pipeline creation wall time; without
  `--cold-compile` the driver cache makes it near zero after the first run.
- A one-workgroup dispatch measures latency, not throughput. Add workgroups (and buffer
  size) to see the shape the game runs at, and test both subgroup sizes: the software HMMA
  fixtures run ~50% slower at wave64 than wave32 on RADV.

## Captured game shaders

The `*-inputs` HMMA fixtures read eight packed half words per invocation after 512 output
bytes, which the `hmma.json` buffer layout matches. The global memory fixtures copy one
element per lane from the input half of buffer `lane % N` to the output half of the next
buffer through the routing helpers, so every candidate buffer is both read and written.

Captured game compute shaders compiled by `shader_sweep.py` (`records/*.spv` in two sweep
outputs) can be paired the same way when `reflect` shows only uniform and storage buffers.
Zero-filled uniform buffers keep loop counts and addresses at zero; give a real constant
buffer image with `"fill": "file"` when the shader needs one, and start with a small
`dispatch`, because a shader that spins on garbage input hangs the GPU until the 30 s fence
timeout.

## Limits

Compute only, with uniform and storage buffer bindings; a module that binds images, samplers
or texel buffers is refused with the offending binding named. Both modules must declare the
same descriptor interface. Repeated dispatches re-run on the same buffers, so a shader that
accumulates with atomics produces different results per repeat; keep `timing` at `0` for it
or compare with `cases` only. Bitwise agreement covers the tested inputs and this driver, not
every input or another vendor's compiler, and a faster isolated dispatch is not a faster
frame: confirm with `run-profile.sh` and `EDEN_NX2_GPU_PROFILE=1` in the same scene.
