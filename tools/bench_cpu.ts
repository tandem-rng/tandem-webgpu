// Throughput of the CPU fills at 2^22 elements, best of five after a warm-up, with the
// runtime's own generators for comparison. Usage: node tools/bench_cpu.ts
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

const rows: [string, number, number][] = [];
const row = (name: string, bytes: number, seconds: number) =>
  rows.push([name, N / seconds / 1e6, bytes / seconds / 2 ** 30]);

const rng = Tandem.seed(42n);
const u32 = new Uint32Array(N), f32 = new Float32Array(N), f64 = new Float64Array(N);

// A reference for what the engine offers: random bytes from the OS and the xorshift of
// Math.random, which the caller must loop over.
row(
  "crypto.getRandomValues, u32",
  4 * N,
  best(() => {
    for (let i = 0; i < N; i += 16384) crypto.getRandomValues(u32.subarray(i, i + 16384));
  }),
);
row(
  "Math.random loop, f64",
  8 * N,
  best(() => {
    for (let i = 0; i < N; i++) f64[i] = Math.random();
  }),
);
row(
  "Math.random bounded loop, range 1000",
  4 * N,
  best(() => {
    for (let i = 0; i < N; i++) u32[i] = Math.floor(Math.random() * 1000);
  }),
);
row(
  "Math.random Box-Muller loop, f64 normal",
  8 * N,
  best(() => {
    for (let i = 0; i < N; i += 2) {
      const r = Math.sqrt(-2 * Math.log(1 - Math.random())), b = 2 * Math.PI * Math.random();
      f64[i] = r * Math.cos(b);
      f64[i + 1] = r * Math.sin(b);
    }
  }),
);
row(
  "Math.random exponential loop, f64",
  8 * N,
  best(() => {
    for (let i = 0; i < N; i++) f64[i] = -Math.log(1 - Math.random());
  }),
);

row("Tandem fillU32", 4 * N, best(() => rng.fillU32(u32)));
row("Tandem fillF32", 4 * N, best(() => rng.fillF32(f32)));
row("Tandem fillF64", 8 * N, best(() => rng.fillF64(f64)));
row("Tandem fillU32Below, range 1000", 4 * N, best(() => rng.fillU32Below(u32, 1000)));
row("Tandem fillNormalF64", 8 * N, best(() => rng.fillNormalF64(f64)));
row("Tandem fillNormalF32", 4 * N, best(() => rng.fillNormalF32(f32)));
row("Tandem fillExponentialF64", 8 * N, best(() => rng.fillExponentialF64(f64)));
row("Tandem fillExponentialF32", 4 * N, best(() => rng.fillExponentialF32(f32)));
row(
  "Tandem nextU32 loop",
  4 * N,
  best(() => {
    for (let i = 0; i < N; i++) u32[i] = rng.nextU32();
  }),
);

const width = Math.max(...rows.map(([name]) => name.length));
console.log(`${"2^22 elements".padEnd(width)}  Melem/s    GiB/s`);
for (const [name, rate, gibs] of rows) {
  console.log(
    `${name.padEnd(width)}  ${rate.toFixed(1).padStart(7)}  ${gibs.toFixed(2).padStart(7)}`,
  );
}
