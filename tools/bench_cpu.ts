// GiB/s of the CPU fills at 2^22 elements, best of five after a warm-up, each next to the
// runtime's own generator. Usage: node tools/bench_cpu.ts
// Run it on a quiet machine: the figures are minimums, but a loaded host still skews them.
import { Tandem } from "../src/mod.ts";

const N = 1 << 22, RUNS = 5;

function best(run: () => void): number {
  run();
  let min = Infinity;
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    run();
    min = Math.min(min, performance.now() - t0);
  }
  return min / 1e3;
}

const rng = Tandem.seed(42n);
const u32 = new Uint32Array(N), f32 = new Float32Array(N), f64 = new Float64Array(N);

// Box-Muller pairs over Math.random, as the Tandem normals.
function boxMuller(z: Float32Array | Float64Array) {
  for (let i = 0; i < N; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - Math.random())), b = 2 * Math.PI * Math.random();
    z[i] = r * Math.cos(b);
    z[i + 1] = r * Math.sin(b);
  }
}

// Each row: the Tandem fill, then a Math.random loop into an array of the same type. Math.random
// is the engine's noncryptographic generator, xorshift128+ in V8.
const rows: [string, Uint32Array | Float32Array | Float64Array, () => void, () => void][] = [
  ["fillU32", u32, () => rng.fillU32(u32), () => {
    for (let i = 0; i < N; i++) u32[i] = Math.random() * 2 ** 32;
  }],
  ["fillF32", f32, () => rng.fillF32(f32), () => {
    for (let i = 0; i < N; i++) f32[i] = Math.random();
  }],
  ["fillF64", f64, () => rng.fillF64(f64), () => {
    for (let i = 0; i < N; i++) f64[i] = Math.random();
  }],
  ["fillU32Below, range 1000", u32, () => rng.fillU32Below(u32, 1000), () => {
    for (let i = 0; i < N; i++) u32[i] = Math.floor(Math.random() * 1000);
  }],
  ["fillNormalF64", f64, () => rng.fillNormalF64(f64), () => boxMuller(f64)],
  ["fillNormalF32", f32, () => rng.fillNormalF32(f32), () => boxMuller(f32)],
  ["fillExponentialF64", f64, () => rng.fillExponentialF64(f64), () => {
    for (let i = 0; i < N; i++) f64[i] = -Math.log(1 - Math.random());
  }],
  ["fillExponentialF32", f32, () => rng.fillExponentialF32(f32), () => {
    for (let i = 0; i < N; i++) f32[i] = -Math.log(1 - Math.random());
  }],
  ["nextU32 loop", u32, () => {
    for (let i = 0; i < N; i++) u32[i] = rng.nextU32();
  }, () => {
    for (let i = 0; i < N; i++) u32[i] = Math.random() * 2 ** 32;
  }],
];

const gibs = (a: { byteLength: number }, seconds: number) =>
  (a.byteLength / seconds / 2 ** 30).toFixed(2).padStart(7);
console.log(`${"2^22 elements, GiB/s".padEnd(26)}  Tandem  baseline`);
for (const [name, out, ours, theirs] of rows) {
  console.log(`${name.padEnd(26)} ${gibs(out, best(ours))}  ${gibs(out, best(theirs))}`);
}
