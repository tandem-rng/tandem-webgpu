// GiB/s of the GPU fill into device memory, no readback. Usage: deno run tools/bench.ts
import { fillBuffer, requestDevice, seed } from "../src/mod.ts";

const device = await requestDevice();
const key = seed(42n);
const n = 1 << 26;

// One buffer for all runs: a fresh buffer is zero-filled by WebGPU, which would be timed too.
const buffer = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.STORAGE });

async function time(count: number): Promise<number> {
  const t0 = performance.now();
  await fillBuffer(device, { key, count, dtype: "u32", buffer });
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3;
}

// Many fills per submit, so the submit and completion latency is amortised.
async function batched(count: number, reps: number): Promise<number> {
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) await fillBuffer(device, { key, count, dtype: "u32", buffer });
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3 / reps;
}

const gibs = (seconds: number) => (4 * n / seconds / 2 ** 30).toFixed(1);
for (let i = 0; i < 5; i++) await time(n);
let single = Infinity, many = Infinity;
for (let i = 0; i < 7; i++) single = Math.min(single, await time(n));
for (let i = 0; i < 5; i++) many = Math.min(many, await batched(n, 32));
console.log(`fill_u32 2^26 words, one fill per submit   ${gibs(single)} GiB/s`);
console.log(`fill_u32 2^26 words, 32 fills per submit   ${gibs(many)} GiB/s`);
