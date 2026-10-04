# Speed

## CPU

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
costs about four times plain arithmetic. The notes below cover that and a WASM comparison.

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

## GPU

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

