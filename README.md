<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .wgsl"></p>

# tandem-webgpu

[Tandem8x32](https://github.com/tandem-rng/spec) for WebGPU: a noncryptographic pseudorandom
number generator built to be fast on CPUs and GPUs alike. A WGSL compute shader fills the
stream on the GPU, and a small TypeScript module derives keys and small draws on the CPU. It
produces the stream the specification defines, bit for bit.

- `tandem.wgsl`: the building blocks `T`, `F`, `F_keyed`, `block`, `split_key`, `sub_key`,
  and the `fill` entry point. WGSL has no 64-bit integers, so the 32x32 to 64 product of the
  mix is built from 16-bit halves, which is exact, and chunk indices travel as word pairs.
- `tandem_f32.wgsl`: the `fill_f32` entry point, which stores Float32 values instead of words.
  It is appended to `tandem.wgsl` when the shader is embedded, so `tandem.wgsl` stays identical
  to the copy in tandem-rs.
- `tandem_derived.wgsl`: the entry points `fill_below32`, `fill_below64` and `normal_pairs` for
  bounded integers and normals (Appendix A of the specification), appended the same way.
- `src/`: the `Tandem` class for draws and fills on the CPU, the key functions `seed`, `split`,
  `sub`, `fork`, the float mappings, and `fill`, which runs the shader and reads back a typed
  array. No dependencies.

## Use

```ts
import {
  fill, fillBelow, fillBuffer, fillMany, fillNormal, fork, requestDevice, seed, split, Tandem,
} from "tandem-webgpu";

const device = await requestDevice();          // a device with the adapter's buffer limits
const key = seed(42n);                          // 128-bit seed through the spec's whitening
const { values, position } = await fill(device, { key, count: 1 << 20, dtype: "f64" });
const next = await fill(device, { key, position, count: 1000, dtype: "u32" });
const worker = split(key, 7n);                  // by index, from the key alone
const { children } = fork(key, position, 4);    // from the current block
const { buffer } = await fillBuffer(device, { key, count: 1 << 24, dtype: "u32" }); // stays on the GPU
const [a, b] = await fillMany(device, [         // one command buffer, one submit
  { key, count: 1 << 20, dtype: "u32" },
  { key: split(key, 1n), count: 1 << 20, dtype: "u32" },
]);
const floats = await fillBuffer(device, { key, count: 1 << 24, dtype: "f32", floats: true }); // f32 values
const dice = await fillBelow(device, { key, count: 1000, dtype: "u32", range: 6 });  // in [0, 6)
const z = await fillNormal(device, { key, count: 1 << 20 });                          // Float32 normals

const rng = Tandem.seed(42n);                   // small draws on the CPU, no device needed
rng.nextF64(); rng.nextU8(); rng.nextBool();    // each aligns to its width, as the spec says
const xs = rng.fillF32(1000);                   // also fillU32, fillU64, fillF64
rng.atU32(5n);                                  // element 5 of the next fill, position unmoved
rng.fillU32Below(10, 6);                        // also fillU64Below, nextU32Below, fillNormalF32/F64
const child = rng.split(7);                     // also sub(purpose) and fork(n), as Tandem objects
```

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

The normals use the device's `log`, `sqrt`, `cos` and `sin` in single precision, with the
angle cut by the nearest quarter turn first so that the builtins only see arguments in
[-pi/4, pi/4]. Implementations may differ in the last places, so values agree with the other
ports to 16 ulps plus 1e-6, the tolerance of Appendix A. The GPU runs the uniform fill and
then a pair pass over the same buffer, so the pairs may straddle stream blocks at any start.
There is no GPU `f64` normal, for lack of an f64 type.

`Tandem` takes a key and an optional `position` and `K`, and exposes `key`, `position` and
`chunkLength`. Its draws are the spec's scalar draws: `nextU8`, `nextU16`, `nextU32`, `nextU64`,
`nextF32`, `nextF64` and `nextBool`. It also has the bounded and normal draws of Appendix A,
`nextU32Below`, `nextU64Below`, `fillU32Below`, `fillU64Below`, `nextNormalF32`,
`nextNormalF64`, `fillNormalF32` and `fillNormalF64`, in double precision with the angle taken
in double and rounded for Float32. Scalar bounded draws reject on the draws that follow, so
only fills decompose by position. Each draw costs one `block` call per 16 bytes, so use
`fill` on the GPU for bulk output.

Deno runs the TypeScript directly. For browsers and Node, `npm run build` emits `dist/`.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Tests

```sh
deno task test
```

`tests/core.test.ts` checks the CPU building blocks against every vector of the specification
(`tests/vectors.json`, a copy of the spec repository's file). It also checks the `Tandem` class:
its fills against every dump in `tests/data` at K = 32 and K = 8, its mixed-width draws, its
fills from mid-stream positions, its derived generators, and the 2^64 position bound. It checks the bounded draws and normals against the fixtures of
tandem-c and tandem-cuda in `tests/cross.json` (`tools/gen_cross.ts` rebuilds it from checkouts
of those repositories), and that a bounded fill cut at arbitrary element boundaries equals the
whole fill at a position with rejections. `tests/gpu.test.ts` checks the
GPU fill against the vectors and against reference stream dumps in `tests/data`
for u32 at K = 32 and K = 8, u64, f32, f64 and u8, from several start positions
and across workgroup boundaries, the signed types against the dump bytes, and `bool` against the
spec's bit vectors and the CPU class from a mid-word start. It also checks that `fillBuffer` with `floats` holds the
mapped values and that without it the buffer keeps raw words. It checks that `fillMany` returns
the same values as single fills for mixed dtypes, positions, K and caller buffers, and that `fill` can reuse a caller buffer across two fills. It checks the GPU bounded fills against the same fixtures and the CPU class across ranges with
many rejections, K and start positions, and the GPU normals against the fixtures to 16 ulps
plus 1e-6, against the CPU class across the two-dimensional pair dispatch, and in `fillMany`.
The GPU tests skip when no adapter exists. CI runs them with
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
| Deno, `fill_normal` f32, 2^26 values, 32 fills per submit | 57 |
| store ceiling: one constant block per invocation, same buffer, 32 per submit | 700 to 720 |

Variants measured on this GPU and rejected, all within noise of the committed shader or
slower: a plain `*` in place of the exact 16-bit-half `mul_hi`, no bounds check, workgroups
of 64 or 128, four steps unrolled per store burst, a lane-major thread mapping, two chunks
per invocation, and a workgroup tile with 512-byte writes per SIMD group, which
measured four times slower. `deno task bench` prints the Deno rows and the ceiling.

A bounded fill costs the same as a plain fill until draws reject. A rejection derives a
fallback key with three seeding functions and then draws on that stream, so the half-rejecting
range above is the worst case and ranges far from 2^31 reject rarely. The normal fill is two
passes, the uniform fill and an in-place pair pass, and runs at about 0.4 of the plain fill.

`demo/index.html` runs the fill in a browser: `python3 -m http.server` in the repo root and
open `/demo/`.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
