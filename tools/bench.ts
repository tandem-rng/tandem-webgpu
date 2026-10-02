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
console.log(`constant store, same buffer, 32 per submit    ${gibs(store)} GiB/s`);
