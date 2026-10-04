// GiB/s of the GPU fill into device memory, no readback. Usage: deno run tools/bench.ts
import { fillBuffer, fillMany, type FillOptions, requestDevice, seed } from "../src/mod.ts";

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

// One submit for `reps` fills of the same kind into the one buffer, through fillMany.
async function submitted(options: Omit<FillOptions, "key" | "count" | "buffer">, reps: number) {
  const items = Array.from({ length: reps }, () => ({ key, count: n, buffer, ...options }));
  const t0 = performance.now();
  await fillMany(device, items);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3 / reps;
}

// The store ceiling: a shader that writes one constant block per invocation into the same
// buffer, timed the same way. The fill is compute bound when it sits well below this.
const CEILING = `@group(0) @binding(0) var<storage, read_write> out: array<vec4<u32>>;
@compute @workgroup_size(256) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
  out[id.x] = vec4<u32>(id.x, 1u, 2u, 3u);
}`;
const ceiling = await device.createComputePipelineAsync({
  layout: "auto",
  compute: { module: device.createShaderModule({ code: CEILING }), entryPoint: "fill" },
});
const bind = device.createBindGroup({
  layout: ceiling.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer } }],
});
async function constant(reps: number): Promise<number> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(ceiling);
  pass.setBindGroup(0, bind);
  for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(n / 4 / 256);
  pass.end();
  const t0 = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3 / reps;
}

const gibs = (seconds: number) => (4 * n / seconds / 2 ** 30).toFixed(1).padStart(6);
for (let i = 0; i < 5; i++) await time(n);
let single = Infinity, many = Infinity, store = Infinity;
for (let i = 0; i < 7; i++) single = Math.min(single, await time(n));
for (let i = 0; i < 7; i++) many = Math.min(many, await batched(n, 32));
for (let i = 0; i < 7; i++) store = Math.min(store, await constant(32));
console.log(`fill_u32 2^26 words, one fill per submit      ${gibs(single)} GiB/s`);
console.log(`fill_u32 2^26 words, 32 fills per submit      ${gibs(many)} GiB/s`);
const derived = {
  "fill_below u32, range 1000": { dtype: "u32", range: 1000 },
  "fill_below u32, range 2^31 + 1": { dtype: "u32", range: 2147483649 },
  "fill_below u64, range 1000": { dtype: "u64", range: 1000n },
  "fill_normal f32": { dtype: "f32", normal: true },
} as const;
for (const [name, options] of Object.entries(derived)) {
  // A u64 word is 8 bytes, so half as many fit the buffer.
  const count = options.dtype === "u64" ? n / 2 : n;
  const items = { ...options, count };
  await submitted(items, 4);
  let best = Infinity;
  for (let i = 0; i < 7; i++) best = Math.min(best, await submitted(items, 32));
  console.log(`${name.padEnd(30)} 2^26 words, 32 per submit  ${gibs(best)} GiB/s`);
}
console.log(`constant store, same buffer, 32 per submit    ${gibs(store)} GiB/s`);
