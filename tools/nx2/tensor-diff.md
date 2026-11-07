# Software matrix GPU comparisons

`tools/nx2/shader_ab.py` (see `shader-ab.md`) runs the same fixtures through the generic
compute A/B runner with reflected bindings, GLSL variants and tabulated timing; `tensor_diff.cpp`
remains for its targeted HMMA rounding, subnormal and overflow input patterns.

`tensor_diff.cpp` runs two emitted software-HMMA shaders on the same Vulkan device
and compares their packed outputs bitwise. Use it before benchmarking changes to
the shader emitter. It requires Vulkan 1.3, compute subgroup shuffles, and a 32- or
64-lane subgroup. It does not load game files or modify the emulator's settings.

Build the runner:

```sh
c++ -std=c++20 -O2 tools/nx2/tensor_diff.cpp -lvulkan -o /tmp/tensor-diff
```

Before modifying the emitter, build the tests and save the baseline fixtures:

```sh
cmake --build build/dev --target tests --parallel 12
EDEN_TENSOR_TEST_OUTPUT=/tmp/tensor-before build/dev/bin/tests '[tensor]'
```

Repeat after the change, using a different output directory:

```sh
cmake --build build/dev --target tests --parallel 12
EDEN_TENSOR_TEST_OUTPUT=/tmp/tensor-after build/dev/bin/tests '[tensor]'
spirv-val --target-env vulkan1.2 /tmp/tensor-after/software-k16-inputs.spv
/tmp/tensor-diff /tmp/tensor-before/software-k16-inputs.spv \
  /tmp/tensor-after/software-k16-inputs.spv --subgroup 32 --cases 1024 --timing 1000
```

Test both K8 and K16, and both subgroup sizes when supported. The `*-inputs`
fixtures accept runtime fragments; the runner supplies deterministic random half
values and targeted rounding, subnormal, sign, and overflow cases. The `*-many`
fixtures exercise dependent matrix calls with fixed inputs. Profile-specific
fixtures can be compared against the corresponding baseline input fixture.

An unsupported device or failed shader operation is an error, not a passing test.
First self-compare a baseline fixture to check the setup. A deliberately different
fixture should fail, confirming that the runner detects changed output.

The optional timing mode alternates baseline/candidate order and reports medians
from GPU timestamps. Each serial dispatch includes a barrier. These small
fixtures measure neither the game's full kernels nor emulator frame rate; confirm
any improvement with the same in-game scene. Bitwise agreement covers the tested
inputs and driver, not every possible input or another driver's behavior.

For live comparisons, `run-profile.sh GAME` launches the GUI directly into a game
with scaling, the local debug server, and profiling enabled. Set
`LICHII_PROFILE=0` for a run without profiling, or `LICHII_BINARY=/path/to/lichii`
to select a preserved comparison build. Close the previous session before
launching another. Keep frame captures separate from timed measurement windows.

## Initial RADV experiments

On the Radeon 8060S with RADV 26.2.5, Lichii PR #174 reduced the measured title-screen
GPU time from approximately 24.7 ms to 19.6 ms. Two follow-up candidates were
tested and dropped:

- Branching around the subnormal rounding calculation preserved the tested
  outputs, but slowed the small K8/K16 GPU fixtures at both subgroup sizes.
- Replacing A's general shuffles with equivalent quad broadcasts preserved the
  tested outputs, but produced effectively unchanged small-fixture timings and
  roughly 19.6–19.8 ms title-screen GPU time.

Both candidates passed 524,288 packed-word comparisons across runtime-input
fixtures. These comparisons establish agreement for those cases, not exhaustive
numerical correctness. The production emitter retains PR #174 without either
candidate. The unprofiled title screen also remained around 42–46 FPS, so tracing
overhead did not explain the remaining gap to 60 FPS.

Keep the game window visible: in this session, screensaver occlusion caused
one-second presentation waits. Exclude those samples from shader comparisons.
