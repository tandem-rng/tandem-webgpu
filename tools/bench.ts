// GiB/s of the GPU fills into device memory, no readback, against a plain Philox4x32-10 shader.
// Usage: deno run tools/bench.ts
import { fillBuffer, fillMany, type FillOptions, requestDevice, seed } from "../src/mod.ts";

const device = await requestDevice();
const key = seed(42n);
const n = 1 << 26;

// One buffer for all runs: a fresh buffer is zero-filled by WebGPU, which would be timed too.
const buffer = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.STORAGE });

async function single(): Promise<number> {
  const t0 = performance.now();
  await fillBuffer(device, { key, count: n, dtype: "u32", buffer });
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3;
}

// One submit for `reps` fills of the same kind into the one buffer, through fillMany.
async function submitted(options: Omit<FillOptions, "key" | "count" | "buffer">, reps: number) {
  const items = Array.from({ length: reps }, () => ({ key, count: n, buffer, ...options }));
  const t0 = performance.now();
  await fillMany(device, items);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1e3 / reps;
}

// The baseline: Philox4x32-10 of Salmon et al. (2011) with the 16-bit-half mul_hi of the Tandem
// shader, since no WebGPU library ships a generator. Invocation t writes blocks t, t + stride, ...,
// block b from counter (b, 0, 0, 0) under key (42, 0).
const PHILOX = `@group(0) @binding(0) var<storage, read_write> out: array<vec4<u32>>;
fn mul_hi(a: u32, b: u32) -> u32 {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let lh = al * bh; let hl = ah * bl;
  let mid = ((al * bl) >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var b = id.x; b < arrayLength(&out); b += nw.x * 256u) {
    var c = vec4<u32>(b, 0u, 0u, 0u);
    var k = vec2<u32>(42u, 0u);
    for (var r = 0u; r < 10u; r++) {
      c = vec4<u32>(mul_hi(0xcd9e8d57u, c.z) ^ c.y ^ k.x, 0xcd9e8d57u * c.z,
                    mul_hi(0xd2511f53u, c.x) ^ c.w ^ k.y, 0xd2511f53u * c.x);
      k += vec2<u32>(0x9e3779b9u, 0xbb67ae85u);
    }
    out[b] = c;
  }
}`;

// The store ceiling: one constant block per invocation into the same buffer. The fill is
// compute bound when it sits well below this.
const CEILING = `@group(0) @binding(0) var<storage, read_write> out: array<vec4<u32>>;
@compute @workgroup_size(256) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
  out[id.x] = vec4<u32>(id.x, 1u, 2u, 3u);
}`;

async function kernel(code: string, workgroups: number) {
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "fill" },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }],
  });
  return async (reps: number) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(workgroups);
    pass.end();
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / 1e3 / reps;
  };
}
const philox = await kernel(PHILOX, 65535);
const ceiling = await kernel(CEILING, n / 4 / 256);

const gibs = (seconds: number) => (4 * n / seconds / 2 ** 30).toFixed(1).padStart(6);
async function best(run: () => Promise<number>): Promise<number> {
  for (let i = 0; i < 3; i++) await run();
  let t = Infinity;
  for (let i = 0; i < 7; i++) t = Math.min(t, await run());
  return t;
}
const philox1 = await best(() => philox(1)), philox32 = await best(() => philox(32));
console.log(`fill_u32, one fill per submit        ${gibs(await best(single))} GiB/s`);
console.log(`Philox4x32-10, one fill per submit   ${gibs(philox1)} GiB/s`);
console.log(
  `fill_u32, 32 per submit              ${
    gibs(await best(() => submitted({ dtype: "u32" }, 32)))
  } GiB/s`,
);
console.log(`Philox4x32-10, 32 per submit         ${gibs(philox32)} GiB/s`);
const derived = {
  "fill_below u32, range 1000": { dtype: "u32", range: 1000 },
  "fill_below u32, range 2^31 + 1": { dtype: "u32", range: 2147483649 },
  "fill_below u64, range 1000": { dtype: "u64", range: 1000n },
  "fill_normal f32": { dtype: "f32", normal: true },
} as const;
for (const [name, options] of Object.entries(derived)) {
  // A u64 word is 8 bytes, so half as many fit the buffer.
  const items = { ...options, count: options.dtype === "u64" ? n / 2 : n };
  console.log(`${name.padEnd(36)} ${gibs(await best(() => submitted(items, 32)))} GiB/s`);
}
console.log(`constant store, 32 per submit        ${gibs(await best(() => ceiling(32)))} GiB/s`);
console.log("all rows 2^26 words, 256 MiB");
