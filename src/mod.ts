// tandem-webgpu: Tandem8x32 for JavaScript runtimes. The full generator on the CPU for Node,
// Deno, Bun and browsers, and fills on the GPU where WebGPU exists.
// Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

export {
  align,
  block,
  checkK,
  DEFAULT_K,
  F,
  fKeyed,
  fork,
  seed,
  split,
  sub,
  T,
  Tandem,
  toFloat32,
  toFloat64,
} from "./core.ts";
export type { Key, State } from "./core.ts";
export { fillCpu } from "./cpu.ts";
export type { CpuFillOptions } from "./cpu.ts";
export {
  fill,
  fillBelow,
  fillBuffer,
  fillExponential,
  fillMany,
  fillNormal,
  requestDevice,
} from "./gpu.ts";
export type { DType, Fill, FillOptions, HostDType, Values } from "./gpu.ts";
