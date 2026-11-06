# Cooperative matrix feasibility probe

This standalone experiment compares the existing software HMMA helper with
`VK_KHR_cooperative_matrix`. It does not modify or launch the emulator. The tested
mapping is **not bitwise equivalent** on Radeon 8060S/RADV and is not a production
optimization.

The host exposes subgroup M16N16K16 with FP16 A/B and FP32 C/result. A guest
M16N8K8 operation is padded to that shape with zero A/B entries. Guest K16 uses
two host operations, with the software helper's explicit half-rounding formula
after each K8 group. The probe stages fragments through shared memory. On a
64-lane host subgroup it processes the two independent guest warps separately;
on 32-lane subgroups each subgroup handles one guest warp.

## Build and run

Requirements: Vulkan 1.3 headers/loader and supported GPU, C++20 compiler,
glslangValidator, spirv-val, and the built Lichii `tests` target. The runner checks
the required matrix shape, compute support, FP16, subgroup size, and Vulkan memory
model, and enables the corresponding device features. No game assets are used.

Run from the repository root:

```sh
c++ -std=c++20 -O2 tools/nx2/cooperative_matrix_probe.cpp -lvulkan -o /tmp/cooperative-matrix-probe
EDEN_TENSOR_TEST_OUTPUT=/tmp/lichii-cooperative-baseline build/dev/bin/tests '[tensor]'
mkdir -p /tmp/lichii-cooperative-results
for depth in 8 16; do
    glslangValidator -V --target-env vulkan1.3 -DGUEST_DEPTH=$depth \
        tools/nx2/cooperative_matrix_probe.comp -o /tmp/cooperative-probe-k$depth.spv
    spirv-val --target-env vulkan1.3 /tmp/cooperative-probe-k$depth.spv
    spirv-val --target-env vulkan1.3 /tmp/lichii-cooperative-baseline/software-k$depth-inputs.spv
    for subgroup in 32 64; do
        /tmp/cooperative-matrix-probe \
            /tmp/lichii-cooperative-baseline/software-k$depth-inputs.spv \
            /tmp/cooperative-probe-k$depth.spv \
            --depth $depth --subgroup $subgroup --cases 1000 --timing 1000 \
            > /tmp/lichii-cooperative-results/k$depth-wave$subgroup.log 2>&1
    done
done
```

Exit 0 means bitwise agreement, 1 means measured differences, and 2 indicates
an unsupported configuration, invalid setup, or failed numerical sanity check.
Exit 1 is expected for the measured mapping. `--device N` selects the enumerated
GPU; the default is 0. Set `--depth` to match both shaders. Each command finishes
its bounded workload and destroys its Vulkan resources.

Input cases use deterministic seed `0x4d4d4100 + case`. Ten repeating categories
cover special half values, random finite values, moderate finite values,
subnormals, halfway rounding, subnormal ties, the minimum-normal boundary,
overflow, identity matrices, and exact dyadic matrices. Identity/dyadic results
are checked against a CPU calculation whose intermediate values are exactly
representable. Those sanity checks accept either sign of zero; the main bitwise
comparison still counts signed-zero differences. Counts refer to packed 32-bit
words, each holding two half values. The reported finite category includes zeros
and subnormals; the normal-finite and signed-zero-only counters distinguish them.

## Measured results, 2026-11-06

Radeon 8060S Graphics, RADV STRIX_HALO; glslang 16.6.0. Both shaders passed
`spirv-val --target-env vulkan1.3`. Each row below compares 1,000 cases and 128,000
packed words, including 200 CPU sanity checks (100 repetitions of each of two
sanity fixtures). All CPU sanity checks passed modulo zero sign.

| Guest depth | Host subgroup | All mismatched words | Normal-finite differences | Signed-zero-only | NaN-payload-only | Software ns/dispatch | Cooperative ns/dispatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| K8 | 32 | 14,655 | 5 | 600 | 10,850 | 489.5 | 2,007.3 |
| K8 | 64 | 14,655 | 5 | 600 | 10,850 | 693.5 | 1,935.9 |
| K16 | 32 | 15,813 | 12 | 0 | 12,600 | 588.3 | 2,933.2 |
| K16 | 64 | 15,813 | 12 | 0 | 12,600 | 1,036.4 | 2,957.5 |

For example, K16 case 2, seed `4d4d4102`, lane 38, output word 1 gives
`4833b439` from the software helper and `4833b43a` from the cooperative probe.
One normal finite half component differs by one representable step. This occurs
on both subgroup sizes and is separate from NaN payload and signed-zero cases.
The half-subnormal tie category also differs in 3,200 packed words per run.

The final observed logs are
`/tmp/lichii-cooperative-results/k{8,16}-wave{32,64}-verified.log`.
The verified runs explicitly enable Vulkan memory-model support required by the
GLSL-generated modules.

The timestamps measure a tiny synthetic dispatch, including shared-memory
staging and barriers. They are medians of ten alternating batches of 1,000 serial
dispatches after warmup. They do not predict game performance or establish a
limit on a differently integrated implementation.

This comparison is against the current software helper, not console hardware.
The GLSL probe and the emulator-emitted SPIR-V also differ in compiler lowering;
no explicit floating-point execution modes are added. Consequently these results
do not isolate the matrix unit as the cause of every difference. Vulkan specifies
implementation-dependent accumulation order and internal precision for
[`OpCooperativeMatrixMulAddKHR`](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
Preserving the K8 boundary alone does not establish arithmetic equivalence.
The observed normal finite differences prevent treating this mapping as an
exact replacement. No production shader path or image-quality shortcut is added.
