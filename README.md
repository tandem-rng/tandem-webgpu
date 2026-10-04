<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .wgsl"></p>

# tandem-webgpu

[Tandem8x32](https://github.com/tandem-rng/spec) for JavaScript: a WGSL shader fills the stream
on the GPU, and the same package runs every draw on the CPU for Node, Deno, Bun and browsers.
The stream is bit exact with the specification, and the CPU and GPU return the same values.

## Install

```sh
npm install github:tandem-rng/tandem-webgpu
```

The install builds `dist/`. In a clone, Deno, Bun and Node 24 or later import `src/mod.ts`
directly. The GPU path needs WebGPU. No dependencies.

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

## What it provides

- `Tandem`: scalar draws, fills into new or caller arrays, and `atU32` random access.
- `fillCpu`: the options and result of `fill`, on the CPU.
- `fill`, `fillBuffer`, `fillMany`: GPU fills of u8 to u64, i8 to i64, f32, f64 and bool.
- `fillBelow`, `nextBelow`: bounded integers by Lemire's method, width chosen from the range.
- `fillNormal`, `fillNormalF32`, `fillNormalF64`: Box-Muller normals, exact with tandem-c on CPU.
- `fillExponential`, `fillExponentialF32`, `fillExponentialF64`: exponentials, same logarithm.
- `seed`, `split`, `sub`, `fork`: keys and child generators.
- Parallel use: a fill cut at any element boundary equals the whole fill (Appendix B).

| Draw | Scalar | Fill |
|---|---|---|
| bool, u8, u16, u32 | `nextBool`, `nextU8`, `nextU16`, `nextU32` | `fillBool`, `fillU8`, `fillU16`, `fillU32` |
| u64 | `nextU64` (BigInt), `nextU64Pair` (`[lo, hi]`) | `fillU64` |
| f32, f64 | `nextF32`, `nextF64` | `fillF32`, `fillF64` |
| bounded | `nextU32Below`, `nextU64Below`, `nextBelow` | `fillU32Below`, `fillU64Below`, `fillBelow` |
| normal | `nextNormalF32`, `nextNormalF64`, `nextNormal2F32`, `nextNormal2F64` | `fillNormalF32`, `fillNormalF64` |
| exponential | `nextExponentialF32`, `nextExponentialF64` | `fillExponentialF32`, `fillExponentialF64` |

Details of every function, the GPU entry points and the CPU design are in
[docs/notes.md](docs/notes.md).

## Tests

```sh
npm test                 # Node 24 or later
bun test tests/core.test.ts tests/cpu.test.ts
deno task test           # also the GPU tests
```

The tests check the spec vectors, the stream dumps in `tests/data`, the cross fixtures below,
cut fills, empty fills, and two hashes of tandem-c's dumps: 1e6 normal pairs at five starts
(SHA-256 `cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded`) and 1e6
exponentials (`5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e`).
`tools/gen_cross.ts` rebuilds `tests/cross.json`. CI runs Node, Deno and Bun.

| Fixture | Commit |
|---|---|
| tandem-c `tests/cross_below.h`, `cross_fill_below.h`, `cross_normal.h`, `cross_exponential.h` | b049384 |
| tandem-cuda `tests/cross_fill_below.h`, `cross_fill_normal.h`, `cross_fill_exponential.h` | c5c5725 |
| the specification's `vectors.json` | f9a74ab |

`cross_normal.h` has SHA-256 `e313b2f1cda2301f8c67cfae952219d4898df9a0623372965c39f6bb0edc7003`.

## Speed

CPU: Apple M4 Pro, Node 26, 2^22 elements, best of five (`npm run bench:cpu`).

| | Melem/s | GiB/s |
|---|---|---|
| `crypto.getRandomValues`, u32 | 3018 | 11.24 |
| `Math.random` loop, f64 | 220 | 1.64 |
| `Math.random` loop, bounded, range 1000 | 224 | 0.83 |
| `Math.random` Box-Muller loop, f64 normal | 61 | 0.46 |
| `Math.random` exponential loop, f64 | 104 | 0.78 |
| `fillU32` | 894 | 3.33 |
| `fillF32` | 596 | 2.22 |
| `fillF64` | 312 | 2.32 |
| `fillU32Below`, range 1000 | 360 | 1.34 |
| `fillNormalF64` | 28 | 0.21 |
| `fillNormalF32` | 36 | 0.13 |
| `fillExponentialF64` | 33 | 0.25 |
| `fillExponentialF32` | 67 | 0.25 |
| `nextU32` loop | 237 | 0.88 |

The normals and exponentials are bit exact with tandem-c, and the emulated fused multiply-add
costs about four times plain arithmetic. See the notes for that and a WASM comparison.

GPU: Apple M4 Pro, `fill_u32` into device memory with no readback, minimum of 7 after a
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

Batch small GPU fills with `fillMany`, since a submit costs about 0.4 ms. `demo/index.html`
runs the fill in a browser.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
