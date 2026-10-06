# Speed

`npm run bench:cpu` produces the CPU figures and `deno task bench` the GPU figures. Both tables
come from one session on an Apple M4 Pro, in GiB/s of output.

## CPU

Node 26, one thread, 2^22 elements, best of five. Each baseline is the fastest generator the
runtime offers for that output: `crypto.getRandomValues` for raw words, otherwise a
`Math.random` loop into an array of the same type, with Box-Muller pairs for the normals and
`-log(1 - u)` for the exponentials.

| Apple M4 Pro | Tandem | baseline |
|---|---|---|
| `fillU32` | 14.6 | 10.8 |
| `fillF32` | 4.62 | 0.84 |
| `fillF64` | 4.88 | 1.64 |
| `fillU32Below`, range 1000 | 1.86 | 0.85 |
| `fillNormalF64` | 1.31 | 0.45 |
| `fillNormalF32` | 0.30 | 0.23 |
| `fillExponentialF64` | 1.01 | 0.77 |
| `fillExponentialF32` | 0.44 | 0.37 |
| `nextU32` loop | 1.07 | 0.87 |

The fills run the WebAssembly SIMD kernel of `src/stream.wat`. The JavaScript kernel, which
engines without WebAssembly SIMD fall back to, fills u32 at about a quarter of that rate. The
float fills map the words in a JavaScript loop, which caps them below the u32 fill.

- The Float64 normals take the exact logarithm only on the 0.43 % of draws that miss the
  ziggurat's inner rectangles, so they keep the emulated fma and stay bit exact with tandem-c.
- The exponentials and Float32 normals round the plain multiply-add. The exact emulation ran
  the Float64 exponentials at a quarter of this rate and the Float32 paths at about half. They
  stay within 4 ulps of tandem-c.
- A 4-wide unrolled map from words to floats measured the same as the plain loop, and a
  16-bit-limb `mulHi` is 1.8 times faster than a double product in the JavaScript lane loop.
- GPU exponentials (`fillExponential`, `exponential_f32`) deviate from the tandem-cuda fixtures by
  at most 0.9 ulp under Deno, within the 8 ulps plus 1e-6 the tests allow.

## GPU

Deno 2.9 (wgpu, naga), 2^26 words into device memory with no readback, minimum of seven after
a warm-up. No WebGPU library ships a generator, so the baseline is a plain Philox4x32-10 shader
in `tools/bench.ts`, one 16-byte block per counter, with the same 16-bit-half `mul_hi` as the
Tandem shader. The derived rows compare with its raw words, marked *.

| Apple M4 Pro GPU | Tandem | Philox4x32-10 |
|---|---|---|
| `fill_u32`, one fill per submit | 17.3 | 16.2 |
| `fill_u32`, 32 fills per submit | 140 | 110 |
| `fill_below` u32, range 1000, 32 per submit | 140 | 110* |
| `fill_below` u32, range 2^31 + 1, 32 per submit | 4.9 | 110* |
| `fill_below` u64, range 1000, 32 per submit | 137 | 110* |
| `fill_normal` f32, 32 per submit | 59.5 | 110* |

A constant 16-byte store per invocation into the same buffer reaches 770 GiB/s, so both
generators are bound by the GPU's integer throughput, not by memory. A submit and its
completion cost about 0.4 ms on wgpu, so a single fill is latency bound: batch fills with
`fillMany` or fill large buffers. Under Chromium (Dawn, Tint) an earlier session filled 167 GiB/s
at 2^26 words with one fill per submit and 213 at 2^24 words with 16 per submit, without a
Philox run beside them.

Variants measured on this GPU and rejected, all within noise of the committed shader or
slower: a plain `*` in place of the exact 16-bit-half `mul_hi`, no bounds check, workgroups
of 64 or 128, four steps unrolled per store burst, a lane-major thread mapping, two chunks
per invocation, and a workgroup tile with 512-byte writes per SIMD group, which
measured four times slower.

A bounded fill costs the same as a plain fill until draws reject. A rejection derives a
fallback key with three seeding functions and then draws on that stream, so the half-rejecting
range above is the worst case and ranges far from 2^31 reject rarely. The normal fill is two
passes, the uniform fill and an in-place pair pass, and runs at about 0.45 of the plain fill.

`demo/index.html` runs the fill in a browser: `python3 -m http.server` in the repo root and
open `/demo/`.
