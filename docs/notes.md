# Notes

Material moved out of the README, unchanged.

## What it provides

- `tandem.wgsl`: the building blocks `T`, `F`, `F_keyed`, `block`, `split_key`, `sub_key`,
  and the `fill` entry point. WGSL has no 64-bit integers, so the 32x32 to 64 product of the
  mix is built from 16-bit halves, which is exact, and chunk indices travel as word pairs.
- `tandem_f32.wgsl`: the `fill_f32` entry point, which stores Float32 values instead of words.
  It is appended to `tandem.wgsl` when the shader is embedded, so `tandem.wgsl` stays identical
  to the copy in tandem-rs.
- `tandem_derived.wgsl`: the entry points `fill_below32`, `fill_below64` and `normal_pairs` for
  bounded integers and normals (Appendix A of the specification), appended the same way.
- `src/`: the `Tandem` class and `fillCpu` for draws and fills on the CPU, the key functions
  `seed`, `split`, `sub`, `fork`, the float mappings, and `fill`, which runs the shader and
  reads back a typed array. No dependencies.

`fill` takes `dtype` `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64` or
`bool`. The signed types read the unsigned words two's complement on the host, and `bool`
returns a Uint8Array of 0 and 1, one stream bit each from `position`. `fillBuffer` and
`fillMany` take the unsigned and float types.

The `fill` entry point writes stream words. `fill` of `f32` runs `fill_f32`, which applies the
spec's mapping `(raw >> 8) * 2^-24` on the GPU, and `fill` of `f64` applies `(raw >> 11) * 2^-53`
on the host. There is no `f64` on the GPU: WGSL has no 64-bit float type, so `f64` stays a host
mapping of the `u64` words. `fillBuffer` returns a storage buffer of whole 16-byte stream blocks
plus the byte offset of the first value. It holds raw words unless you pass `floats: true` with
`dtype: "f32"`, which stores Float32 values for a later GPU stage. Pass `buffer` to write into
your own. `fill` leaves a buffer you pass alive and destroys only the buffers it creates.

`fillMany(device, items)` takes an array of the same options, encodes every dispatch into one
compute pass and submits once, then returns one result per item in order. It checks all items
before it submits, and items run in order, so they may share a buffer.

A fill larger than the adapter's `maxStorageBufferBindingSize` needs no special call. Every fill
is exact at any block boundary, so `fillBuffer`, `fill`, `fillBelow`, `fillNormal` and `fillMany`
bind the buffer in windows of at most that size, at offsets that meet the offset alignment, and
dispatch each one. The normal pass cuts at even elements. The buffer itself must still fit
`maxBufferSize`, and a fill that needs more throws a `RangeError`.

Bounded integers and normals follow Appendix A of the specification. `fillBelow(device, {...,
dtype: "u32" | "u64", range})` returns values in `[0, range)` by Lemire's method. Element `i`
takes draw `i`, and a draw that Lemire rejects retries on a fallback generator keyed by the
draw's index `g` in the stream, `split(g)` of `sub(P_w)` of the key, so a fill cut at any
element boundary equals the whole fill. A range of 0 gives 0, and an empty fill leaves the
position unchanged. `fillNormal(device, {key, position, count})` returns Float32 standard
normals by Box-Muller: elements `2j` and `2j + 1` come from uniform draws `2j` and `2j + 1`, so
an odd count consumes one draw more than it writes. Both are also options of `fillBuffer`,
`fillMany` and `fill`: `range` for the integers and `normal: true` with `dtype: "f32"`. A
caller `buffer` for a normal fill must hold the blocks of the draws it consumes.

The GPU normals are computed in single precision with the arithmetic of tandem-c: `log` is a
short series on the exponent-split argument and `cos` and `sin` are Taylor series on an angle cut
by the nearest quarter turn, so only `fma` and `sqrt` come from the device. WGSL leaves the
accuracy of its own `log`, `cos` and `sin` open and a software rasteriser misses the tolerance
with them. Values agree with the other ports to 16 ulps plus 1e-6, the tolerance of Appendix A.
The GPU runs the uniform fill and then a pair pass over the same buffer, so the pairs may
straddle stream blocks at any start. There is no GPU `f64` normal, for lack of an f64 type.

Deno and Bun run the TypeScript directly, and so does Node 24 or later. For browsers and older
Node, `npm run build` emits `dist/`.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

### CPU path

`Tandem` takes a key and an optional `position` and `K`, and exposes `key`, `position` and
`chunkLength`. It gives every draw the shaders give, with the same values, on any runtime.
Each fill takes a count and returns a new typed array, or takes a typed array, which may be a
`subarray`, and fills it in place. Fills move the position past the draws they consume. An
empty plain fill returns the aligned position, and an empty bounded, normal or exponential fill
moves nothing.

| Draw | Scalar | Fill | Array |
|---|---|---|---|
| bool | `nextBool` | `fillBool` | `Uint8Array` of 0 and 1, one stream bit each |
| u8, u16 | `nextU8`, `nextU16` | `fillU8`, `fillU16` | `Uint8Array`, `Uint16Array` |
| u32 | `nextU32` | `fillU32` | `Uint32Array` |
| u64 | `nextU64` as BigInt, `nextU64Pair` as `[lo, hi]` | `fillU64` | `BigUint64Array`, whose bytes are the pairs |
| f32, f64 | `nextF32`, `nextF64` | `fillF32`, `fillF64` | `Float32Array`, `Float64Array` |
| bounded | `nextU32Below`, `nextU64Below`, `nextBelow` | `fillU32Below`, `fillU64Below`, `fillBelow` | `Uint32Array`, `BigUint64Array` |
| normal | `nextNormalF32`, `nextNormalF64`, `nextNormal2F32`, `nextNormal2F64` | `fillNormalF32`, `fillNormalF64` | `Float32Array`, `Float64Array` |
| exponential | `nextExponentialF32`, `nextExponentialF64` | `fillExponentialF32`, `fillExponentialF64` | `Float32Array`, `Float64Array` |

The bounded draws follow Appendix A. A scalar draw rejects on the draws that follow, and a fill
retries a rejected element on the fallback generator of its global draw index, so a fill cut at
any element boundary equals the whole fill. `fillU32Below` and `fillU64Below` name the draw
width. `nextBelow` and `fillBelow` take it from the range, 32 bits up to 2^32 and 64 bits above,
as Appendix A asks of an interface that names only the result type, so a bigint range of 1000
gives the values of a `u32` fill in a `BigUint64Array`. A range of 0 gives 0, and a number range
reaches 2^32. `nextNormal2F64` returns the pair of a Box-Muller step, and a normal fill is the
flattened sequence of those pairs: an odd count writes the cosine half of its last pair and
still consumes both uniforms. Exponentials are `-log(1 - u)` of one uniform each.

Normals and exponentials copy the polynomials of tandem-c with the same operation order, and
JavaScript has no fused multiply-add, so `fma64` and `fma32` emulate it exactly: Dekker's product
and a two-term sum, with the one rounding tie the double sum can hit settled by its lost part.
The values equal tandem-c bit for bit on every engine. The emulation costs about four times the
speed of plain multiply and add. Plain arithmetic would change 4 % of the Float64 normals and
0.1 % of the Float64 exponentials, by up to 5.5e-16 relative, so the exact form is the only one.

`fillCpu(options)` takes the options of `fill`, with `normal` also for `f64` and `exponential`
for `f32` and `f64`, and returns `{ values, position }` synchronously. It covers every `dtype`
of `fill`, the signed types and `bool` included, so code can pick the CPU or the GPU path and
read the same result.

The fills run the stream kernel of `src/stream.ts`: the eight lanes of a chunk group stepped row
by row with the lane state in locals, writing the words straight into the array. Scalar draws
read a window of eight rows ahead and step on from the last window of a chunk group without
reseeding. The `atU32`, `atU64`, `atF32` and `atF64` reads compute one block each, which suits a
single element at a far position.

Fixtures of the other ports, in `tests/cross.json` (`tools/gen_cross.ts` rebuilds it):

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


## Tests

```sh
npm test                 # Node 24 or later
bun test tests/core.test.ts tests/cpu.test.ts
deno task test           # also the GPU tests
```

`tests/core.test.ts` and `tests/cpu.test.ts` use `node:test` and run on Node, Deno and Bun, and
CI runs all three. `tests/core.test.ts` checks the CPU building blocks against every vector of
the specification (`tests/vectors.json`, a copy of the spec repository's file). It also checks
the `Tandem` class: its fills against every dump in `tests/data` at K = 32 and K = 8, its
mixed-width draws, its fills from mid-stream positions, its derived generators, and the 2^64
position bound. It checks the bounded draws and normals bit for bit against the fixtures above,
and that a bounded fill cut at arbitrary element boundaries equals the whole fill at a position
with rejections.

`tests/cpu.test.ts` checks the exponentials against the fixtures of tandem-c and tandem-cuda
and the two hashes above. It checks the emulated `fma64` and `fma32` against an exact BigInt
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


## Speed

The fill is compute bound on Apple GPUs at about 160 to 210 GiB/s, under Chromium's Tint and
under Deno's naga alike. The store ceiling of the same buffer and dispatch
shape is 720 GiB/s. A submit and its completion cost about 0.4 ms on wgpu, so a single small
fill is latency bound: batch fills with `fillMany` or fill large buffers.

Apple M4 Pro GPU, `fill_u32` into device memory with no readback, minimum of 7 after a
warm-up:

| | GiB/s |
|---|---|
| Chromium (Dawn, Tint), 2^26 and 2^27 words, one fill per submit | 167 |
| Chromium, 2^24 words, 16 fills per submit | 213 |
| Deno 2.9 (wgpu, naga), 2^26 words, 32 fills per submit | 134 to 144 |
| Deno 2.9, 2^26 words, one fill per submit | 18.5 |
| Deno, `fill_below` u32, range 1000, 2^26 words, 32 fills per submit | 131 |
| Deno, `fill_below` u64, range 1000, 2^26 words (2^25 values), 32 fills per submit | 128 |
| Deno, `fill_below` u32, range 2^31 + 1 (half of the draws reject), 32 fills per submit | 5 |
| Deno, `fill_normal` f32, 2^26 values, 32 fills per submit | 62 |
| store ceiling: one constant block per invocation, same buffer, 32 per submit | 700 to 720 |

Variants measured on this GPU and rejected, all within noise of the committed shader or
slower: a plain `*` in place of the exact 16-bit-half `mul_hi`, no bounds check, workgroups
of 64 or 128, four steps unrolled per store burst, a lane-major thread mapping, two chunks
per invocation, and a workgroup tile with 512-byte writes per SIMD group, which
measured four times slower. `deno task bench` prints the Deno rows and the ceiling.

A bounded fill costs the same as a plain fill until draws reject. A rejection derives a
fallback key with three seeding functions and then draws on that stream, so the half-rejecting
range above is the worst case and ranges far from 2^31 reject rarely. The normal fill is two
passes, the uniform fill and an in-place pair pass, and runs at about 0.45 of the plain fill.

`demo/index.html` runs the fill in a browser: `python3 -m http.server` in the repo root and
open `/demo/`.


### CPU speed notes

Measured on the Apple M4 Pro with Node 26 at 2^22 elements, best of five.

- Plain multiply and add in place of the emulated fma runs Float64 normals at 111 Melem/s,
  Float64 exponentials at 157 and Float32 normals at 150, about four times the exact form. It
  changes 4 % of the Float64 normals and 0.1 % of the Float64 exponentials, by up to 5.5e-16
  relative, and none of the Float32 normals in 4M samples.
- A WASM build of tandem-c (zig cc, wasm32-wasi, -O2 -msimd128, run under Node) fills 3700 Melem/s
  of u32, 1240 of f64 and 1290 of bounded u32, about four times this package. Its normals and
  exponentials run at 9.6 and 12.5 Melem/s, slower than here, since WASM has no fma and the
  library call is slow. It is not part of this package.
- A 4-wide unrolled map from words to floats measured the same as the plain loop, and
  a 16-bit-limb `mulHi` is 1.8 times faster than a double product in the lane loop.
- GPU exponentials (`fillExponential`, `exponential_f32`) deviate from the tandem-cuda fixtures by
  at most 0.9 ulp on the Apple M4 Pro under Deno, within the 8 ulps plus 1e-6 the tests allow.
