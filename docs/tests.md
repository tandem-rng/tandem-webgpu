# Tests

```sh
npm test                 # Node 24 or later
bun test tests/core.test.ts tests/cpu.test.ts
deno task test           # also the GPU tests
```

## Fixtures


| Fixture | Commit |
|---|---|
| tandem-c `tests/cross_below.h`, `cross_fill_below.h`, `cross_normal.h`, `cross_exponential.h` | b049384 |
| tandem-cuda `tests/cross_fill_below.h`, `cross_fill_normal.h`, `cross_fill_exponential.h` | c5c5725 |
| the specification's `vectors.json` | f9a74ab |

`cross_normal.h` has SHA-256 `e313b2f1cda2301f8c67cfae952219d4898df9a0623372965c39f6bb0edc7003`.
The 1e6-pair normal dump of tandem-c's `tools/dump_normals.c`, at starts 0, 1, 77, 12345 and
2^30, hashes to `cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded`, and the
exponential dump of `tools/dump_exponentials.c` to
`5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e`. The tests reproduce both.


## Suites

`tests/core.test.ts` and `tests/cpu.test.ts` use `node:test` and run on Node, Deno and Bun, and
CI runs all three. `tests/core.test.ts` checks the CPU building blocks against every vector of
the specification (`tests/vectors.json`, a copy of the spec repository's file). It also checks
the `Tandem` class: its fills against every dump in `tests/data` at K = 32 and K = 8, its
mixed-width draws, its fills from mid-stream positions, its derived generators, and the 2^64
position bound. It checks the bounded draws and normals bit for bit against the fixtures,
and that a bounded fill cut at arbitrary element boundaries equals the whole fill at a position
with rejections.

`tests/cpu.test.ts` checks the exponentials against the fixtures of tandem-c and tandem-cuda
and the two hashes in Fixtures. It checks the emulated `fma64` and `fma32` against an exact BigInt
oracle on random and on halfway cases, and the Horner steps of the three polynomials against the
chain of exact fused steps. It checks that scalar draws equal the fills across rows, windows and
chunk groups for K = 1, 8 and 32, from every bit offset for u8, u16 and bool, and that fills
equal the `block` function at K = 1 and K = 65536. It checks fills into a caller array or
`subarray`, empty fills of every kind, the width rule of `nextBelow` and `fillBelow`, normal
and exponential fills cut at element boundaries, the 2^64 bound for draws and fills, and
`fillCpu`.

`tests/gpu.test.ts` checks the GPU fill against the vectors and against reference stream dumps
in `tests/data` for u32 at K = 32 and K = 8, u64, f32, f64 and u8, from several start positions
and across workgroup boundaries, the signed types against the dump bytes, and `bool` against the
spec's bit vectors and the CPU class from a mid-word start. It also checks that `fillBuffer` with `floats` holds the
mapped values and that without it the buffer keeps raw words. It checks that `fillMany` returns
the same values as single fills for mixed dtypes, positions, K and caller buffers, and that `fill` can reuse a caller buffer across two fills. It checks the GPU bounded fills against the same fixtures and the CPU class across ranges with
many rejections, K and start positions, and the GPU normals against the fixtures to 16 ulps
plus 1e-6, against the CPU class across the two-dimensional pair dispatch, and in `fillMany`.
It checks that `fillCpu` equals the GPU fill for every dtype and for bounded fills, and the
normals within that tolerance. It checks that every dtype, bounded and normal fill, and
`fillMany`, under a binding limit shrunk to a few windows, equals the unchunked fill, and that a
fill just over the real limit of a software adapter equals the CPU class at the window boundary
and the end. The GPU tests skip when no adapter exists. CI runs them with
a software Vulkan adapter on Linux and on the macOS runner's GPU, validates the shader with
`naga`, and fails when the embedded shader or the vectors drift.

