# Tests

```sh
npm test                 # Node 24 or later
bun test tests/core.test.ts tests/cpu.test.ts tests/conformance.test.ts
deno task test           # also the GPU tests
```

## Suite

`tests/core.test.ts`, `tests/cpu.test.ts` and `tests/conformance.test.ts` use `node:test` and
run on Node, Deno and Bun.

`tests/conformance.test.ts` reads the conformance files of the specification in
`tests/conformance` and has one test per section of the spec's `conformance/CHECKLIST.md` at
2a4bd08. It
runs every case of `below.json`, `fill_below.json`, `normal.json`, `exponential.json` and
`choice.json` on the `Tandem` class and checks the values and the end position, the seven empty
fills included. It checks that `n` scalar draws equal each Float64 normal, exponential and
choice fill, with the same end. It checks that a start one draw later continues the same global
fill for the bounded, normal and choice cases named in the checklist, the width a bounded draw
takes from its range, range 0, the end position of an odd Float32 normal fill, and the pair rule
of Box-Muller. It builds the table of every choice case and compares `capacity`, `cut` and
`alias`, checks a scalar choice, `m = 1` and the rejected weights, and the law of 1e6 choice
draws by chi-square. It cuts every fill case at elements 1, 7, 20, 21 and `n - 1`, and the
Float32 normals at the pair boundaries 2, 8, 20 and the largest even element below `n`, and
fills the pieces in order on one generator. The GPU suite cuts at the same elements. It reproduces the
stream hashes of `hashes.json` for every type the port fills, checks the dumps in `tests/data`
against them, and reproduces the dump hashes of the Float64 and Float32 normals. It checks the
2^63 start bound and a UInt64 draw at 2^63 - 1.

`tests/core.test.ts` checks the CPU building blocks against every vector of the specification
(`tests/vectors.json`, a copy of the spec repository's file). It also checks the `Tandem` class:
its mixed-width draws, its fills and `atU32` from mid-stream positions across chunk groups, its
derived generators, and a bounded fill cut at arbitrary boundaries at K = 32 and K = 8 with
rejections.

`tests/cpu.test.ts` checks the exponentials and Float32 normals on the 1e6-element fills of the
exponential dump: the Float64 exponentials against the exact fused form within 4 ulps, whose
FNV-1a with the Float32 exponentials equals the dump of tandem-c, which makes the Float32
exponentials bit exact, the Float32 exponentials within 0.58 ulp and the normals against libm on the
same uniforms. It checks the first four moments and the KS law of 1e7 draws of each normal and
exponential. It checks the emulated `fma64` against an exact BigInt oracle on random and on
halfway cases, and the Horner steps of the logarithm's polynomial against the chain of exact
fused steps. It checks that the JavaScript kernel gives the words of the WebAssembly kernel
across chunk groups and output buffers. It checks that scalar draws equal the fills across rows,
windows and chunk groups for K = 1, 8 and 32, from every bit offset for u8, u16 and bool, and
that fills equal the `block` function at K = 1 and K = 65536. It checks fills into a caller array
or `subarray`, empty fills of every kind, the width rule of `nextBelow` and `fillBelow`, normal
and exponential fills cut at element boundaries, a Float64 normal fill cut at a missed element,
a choice of 2^20 + 3 columns against the 128-bit products of Appendix C, and `fillCpu`.

`tests/gpu.test.ts` checks the GPU fill against the vectors and against reference stream dumps
in `tests/data` for u32 at K = 32 and K = 8, u64, f32, f64 and u8, from several start positions
and across workgroup boundaries, the signed types against the dump bytes, and `bool` against the
spec's bit vectors and the CPU class from a mid-word start. It also checks that `fillBuffer` with
`floats` holds the mapped values and that without it the buffer keeps raw words. It checks that
`fillMany` returns the same values as single fills for mixed dtypes, positions, K and caller
buffers, and that `fill` can reuse a caller buffer across two fills. It runs every bounded,
Float32 normal, Float32 exponential and choice case of the conformance files on the GPU, values
and end position, and cut at the elements of the checklist. It checks the GPU bounded fills
against the CPU class across ranges with many rejections, K and start positions, the GPU normals
against the CPU class across the two-dimensional pair dispatch, a choice of 2^20 + 3 columns,
whose draws carry into the column index, and derived and choice fills in `fillMany`. It checks
the 2^63 start bound and fills that cross 2^63. It checks that `fillCpu` equals the GPU fill for
every dtype, for bounded fills and for choice, and the normals and exponentials within their
tolerance. It checks that every dtype, bounded, normal and choice fill, and `fillMany`, under a
binding limit shrunk to a few windows, equals the unchunked fill, and that a fill just over the
real limit of a software adapter equals the CPU class at the window boundary and the end. The GPU
tests skip when no adapter exists.

## Checklist gaps

The tests state what the port does instead of each item its API can not express.

- Complex draws and the UInt128, Float16 and Char fills do not exist in this port, so their
  stream hashes and the complex draw across a block do not apply.
- A start lies below 2^63 and a count is a number below 2^53, so no fill can reach 2^64. The
  endpoint check stays in the code but no test can reach it.
- The Float64 exponentials round each multiply-add twice for speed and stay within 4 ulps of
  the fixtures. Their exact fused form matches the dump hash.

## Fixtures

| Fixture | Commit |
|---|---|
| the specification's `conformance/*.json`, copied byte for byte to `tests/conformance` | 2a4bd08 |
| the specification's `vectors.json` | 2a4bd08 |
| the specification's `tables/normal_f64_zig1024.json`, SHA-256 checked | 469a0ae |

The conformance files come from tandem-c at 1c75956. `tests/data` holds six of the stream dumps
of tandem-c, whose SHA-256 the tests check against `hashes.json`.

## CI

- CI runs `tests/core.test.ts`, `tests/cpu.test.ts` and `tests/conformance.test.ts` on Node,
  Deno and Bun.
- CI runs the GPU tests with a software Vulkan adapter on Linux and on the macOS runner's GPU.
- CI validates the shader with `naga`, and fails when the embedded shader, the embedded
  WebAssembly kernel, the vectors, the conformance files or the ziggurat tables drift from the
  spec at the pinned commit.
