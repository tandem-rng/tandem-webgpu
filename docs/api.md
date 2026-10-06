# API

## Use

```ts
import { fill, fillCpu, requestDevice, seed, split, Tandem } from "tandem-webgpu";

const key = seed(42n);
const rng = new Tandem(key);
const xs = rng.fillF64(1 << 20);          // or rng.fillF64(out) to fill your own array
const dice = rng.fillU32Below(1000, 6);
const z = rng.fillNormalF64(1000);        // fillExponentialF64 likewise
const worker = split(key, 7n);            // also sub(purpose), fork(n)

const cpu = fillCpu({ key, count: 1000, dtype: "u32" });             // { values, position }
const gpu = await fill(await requestDevice(), { key, count: 1000, dtype: "u32" });
```

## Reference

- `Tandem`: scalar draws, fills into new or caller arrays, and `atU32` random access.
- `fillCpu`: the options and result of `fill`, on the CPU.
- `fill`, `fillBuffer`, `fillMany`: GPU fills of u8 to u64, i8 to i64, f32, f64 and bool.
- `fillBelow`, `nextBelow`: bounded integers by Lemire's method, width chosen from the range.
- `fillNormal`, `fillNormalF32`, `fillNormalF64`: Box-Muller f32 and ziggurat f64 normals, exact
  with tandem-c on CPU.
- `fillExponential`, `fillExponentialF32`, `fillExponentialF64`: exponentials, same logarithm.
- `seed`, `split`, `sub`, `fork`: keys and child generators.

## Shaders and GPU fills

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

## CPU path

`Tandem` takes a key and an optional `position` and `K`, and exposes `key`, `position` and
`chunkLength`. It gives every draw the shaders give, with the same values, on any runtime.
Each fill takes a count and returns a new typed array, or takes a typed array, which may be a
`subarray`, and fills it in place. Fills move the position past the draws they consume. An
empty plain or Float64 normal fill returns the aligned position, and an empty bounded, Float32
normal or exponential fill moves nothing.

| Draw | Scalar | Fill | Array |
|---|---|---|---|
| bool | `nextBool` | `fillBool` | `Uint8Array` of 0 and 1, one stream bit each |
| u8, u16 | `nextU8`, `nextU16` | `fillU8`, `fillU16` | `Uint8Array`, `Uint16Array` |
| u32 | `nextU32` | `fillU32` | `Uint32Array` |
| u64 | `nextU64` as BigInt, `nextU64Pair` as `[lo, hi]` | `fillU64` | `BigUint64Array`, whose bytes are the pairs |
| f32, f64 | `nextF32`, `nextF64` | `fillF32`, `fillF64` | `Float32Array`, `Float64Array` |
| bounded | `nextU32Below`, `nextU64Below`, `nextBelow` | `fillU32Below`, `fillU64Below`, `fillBelow` | `Uint32Array`, `BigUint64Array` |
| normal | `nextNormalF32`, `nextNormalF64`, `nextNormal2F32` | `fillNormalF32`, `fillNormalF64` | `Float32Array`, `Float64Array` |
| exponential | `nextExponentialF32`, `nextExponentialF64` | `fillExponentialF32`, `fillExponentialF64` | `Float32Array`, `Float64Array` |

The bounded draws follow Appendix A. A scalar draw rejects on the draws that follow, and a fill
retries a rejected element on the fallback generator of its global draw index, so a fill cut at
any element boundary equals the whole fill. `fillU32Below` and `fillU64Below` name the draw
width. `nextBelow` and `fillBelow` take it from the range, 32 bits up to 2^32 and 64 bits above,
as Appendix A asks of an interface that names only the result type, so a bigint range of 1000
gives the values of a `u32` fill in a `BigUint64Array`. A range of 0 gives 0, and a number range
reaches 2^32.

Float64 normals are the 1024-layer ziggurat of Appendix A. Element `i` takes 64-bit draw `i`.
A draw that misses the inner rectangles, 0.43 % of them, continues on a fallback generator keyed
by its global draw index `g`, `split(g)` of `sub(0x4e524d3634)` of the key, and leaves the
position alone. So a scalar draw consumes one draw, and a fill cut at any element equals the
whole fill. `tools/gen_zig_tables.ts` writes the tables of `src/zig_tables.ts` from the spec's
`tables/normal_f64_zig1024.json`. `nextNormal2F32` returns the pair of a Box-Muller step, and a
Float32 normal fill is the flattened sequence of those pairs: an odd count writes the cosine half
of its last pair and still consumes both uniforms. Exponentials are `-log(1 - u)` of one uniform
each.

Normals and exponentials copy the polynomials of tandem-c with the same operation order, and
JavaScript has no fused multiply-add, so `fma64` and `fma32` emulate it exactly: Dekker's product
and a two-term sum, with the one rounding tie the double sum can hit settled by its lost part.
The values equal tandem-c bit for bit on every engine. The emulation costs about four times the
speed of plain multiply and add. Plain arithmetic would change 0.1 % of the Float64
exponentials, by up to 5.5e-16 relative, so the exact form is the only one.

`fillCpu(options)` takes the options of `fill`, with `normal` also for `f64` and `exponential`
for `f32` and `f64`, and returns `{ values, position }` synchronously. It covers every `dtype`
of `fill`, the signed types and `bool` included, so code can pick the CPU or the GPU path and
read the same result.

The fills run the stream kernel of `src/stream.wat`, WebAssembly SIMD that steps the eight lanes
of a chunk group as two vectors of four, and copies 256 rows at a time into the array. Where an
engine lacks WebAssembly SIMD, or a content security policy forbids compiling it, the JavaScript
kernel of `src/stream.ts` gives the same words at a quarter of the speed. Scalar draws read a
window of eight rows ahead and step on from the last window without reseeding. The `atU32`, `atU64`, `atF32` and `atF64` reads compute one block each, which suits a
single element at a far position.

## Parallel use

A fill cut at any element boundary equals the whole fill. Element `i` of a fill is draw `i`, so
ranks, threads or devices that start at the position of their first element, or draw from
`split(task)`, reproduce a serial run for any decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
