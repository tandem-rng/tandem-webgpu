// `fillCpu`: the options and the result of the GPU `fill`, computed on the CPU, for runtimes
// without WebGPU. Node, Deno, Bun and browsers get the same values the shaders give.

import { DEFAULT_K, type Key, Tandem } from "./core.ts";
import type { Fill, FillOptions, HostDType, Values } from "./gpu.ts";

export type CpuFillOptions<D extends HostDType> =
  & Omit<FillOptions, "buffer" | "floats" | "dtype">
  & { dtype: D };

const WIDTH = {
  bool: 1,
  u8: 8,
  i8: 8,
  u16: 16,
  i16: 16,
  u32: 32,
  i32: 32,
  f32: 32,
  u64: 64,
  i64: 64,
  f64: 64,
} as const;

/**
 * Fill `count` values of `dtype` as `count` scalar draws would, on the CPU. The options are
 * those of `fill`, and `normal` and `exponential` also take `f64`, which the GPU lacks. The
 * signed types read the unsigned words two's complement and `bool` gives one stream bit per
 * element. The values equal the GPU's for every dtype and for bounded integers, and for
 * normals within the tolerance of Appendix A.
 */
export function fillCpu<D extends HostDType>(
  { key, position = 0n, count, dtype, K = DEFAULT_K, range, normal, exponential }: CpuFillOptions<
    D
  >,
): Fill<Values<D>> {
  const w = WIDTH[dtype];
  const float = dtype === "f32" || dtype === "f64";
  if ((normal || exponential) && !float) {
    throw new RangeError("normal and exponential apply to dtype f32 and f64 only");
  }
  if (normal && exponential) throw new RangeError("normal and exponential exclude each other");
  if (range !== undefined && (dtype !== "u32" && dtype !== "u64" || normal || exponential)) {
    throw new RangeError("range applies to dtype u32 and u64 only");
  }
  const bound = range === undefined ? 0n : BigInt(range);
  if (bound < 0n || bound >> BigInt(w)) throw new RangeError(`range must fit in ${w} bits`);

  const rng = new Tandem(key as Key, { position, K });
  const values = (() => {
    if (range !== undefined) {
      return dtype === "u32"
        ? rng.fillU32Below(count, Number(bound))
        : rng.fillU64Below(count, bound);
    }
    switch (dtype) {
      case "bool":
        return rng.fillBool(count);
      case "u8":
        return rng.fillU8(count);
      case "i8":
        return new Int8Array(rng.fillU8(count).buffer);
      case "u16":
        return rng.fillU16(count);
      case "i16":
        return new Int16Array(rng.fillU16(count).buffer);
      case "u32":
        return rng.fillU32(count);
      case "i32":
        return new Int32Array(rng.fillU32(count).buffer);
      case "u64":
        return rng.fillU64(count);
      case "i64":
        return new BigInt64Array(rng.fillU64(count).buffer);
      case "f32":
        return normal
          ? rng.fillNormalF32(count)
          : exponential
          ? rng.fillExponentialF32(count)
          : rng.fillF32(count);
      case "f64":
        return normal
          ? rng.fillNormalF64(count)
          : exponential
          ? rng.fillExponentialF64(count)
          : rng.fillF64(count);
    }
  })();
  return { values: values as Values<D>, position: rng.position };
}
