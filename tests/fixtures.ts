// Readers for the conformance files of the specification. tests/conformance holds byte copies
// of tandem-spec conformance/*.json at 2a4bd08, the commit CI pins.
import { createHash } from "node:crypto";
import { align, type Key } from "../src/mod.ts";
import below from "./conformance/below.json" with { type: "json" };
import choice from "./conformance/choice.json" with { type: "json" };
import exponential from "./conformance/exponential.json" with { type: "json" };
import fillBelow from "./conformance/fill_below.json" with { type: "json" };
import hashJson from "./conformance/hashes.json" with { type: "json" };
import normal from "./conformance/normal.json" with { type: "json" };

type Stream = {
  file: string;
  key: string[];
  K: number;
  start: number;
  type: string;
  n: number;
  sha256: string;
};
type Dump = {
  id: string;
  key: string[];
  K: number;
  starts: number[];
  draws: { kind: string; n: number }[];
  fnv1a: string;
  sha256?: string;
  end?: number;
};
export const hashes = hashJson as { streams: Stream[]; dumps: Dump[] };

type Raw = {
  id: string;
  kind: string;
  key: string[];
  K: number;
  start: number;
  n: number;
  values: string[];
  range?: string;
  weights?: string[];
  capacity?: string;
  cut?: string[];
  alias?: string[];
  end?: number;
  rejected?: number;
  tol?: unknown;
};

export const hex32 = (h: string) => parseInt(h, 16) >>> 0;
export const keyOf = (words: string[]) => words.map(hex32) as unknown as Key;
const u64 = (h: string) => BigInt(`0x${h}`);
export const f64 = (h: string) => new Float64Array(BigUint64Array.of(u64(h)).buffer)[0];
const f32 = (h: string) => new Float32Array(Uint32Array.of(hex32(h)).buffer)[0];

const files: Record<string, { cases: Raw[] }> = {
  below,
  fill_below: fillBelow,
  normal,
  exponential,
  choice,
};

/** The cases of one file, by file name without `.json`. */
export const cases = (file: keyof typeof files) => files[file].cases;
export const allCases = () => Object.values(files).flatMap((f) => f.cases);

/** The case whose `id` ends with `name`, such as `CROSS_BELOW32[4]`. */
export function byName(name: string): Raw {
  const found = allCases().filter((c) => c.id.endsWith(` ${name}`));
  if (found.length !== 1) throw new Error(`${found.length} cases named ${name}`);
  return found[0];
}

/** The draw width of a kind. */
export const widthOf = (kind: string) => kind.endsWith("u32") || kind.endsWith("f32") ? 32 : 64;

/** The expected values as the typed array the CPU and GPU fills return. */
export function expected(c: Raw): Uint32Array | BigUint64Array | Float32Array | Float64Array {
  if (c.kind === "fill_choice" || c.kind.endsWith("u32")) return Uint32Array.from(c.values, hex32);
  if (c.kind.endsWith("u64")) return BigUint64Array.from(c.values, u64);
  if (c.kind.endsWith("f32")) return Float32Array.from(c.values, f32);
  return Float64Array.from(c.values, f64);
}

/** The position after the case: its `end`, else the rule of Appendix A or C for its kind. A
 * Float32 normal fill consumes both draws of its last pair. */
export function endOf(c: Raw): bigint {
  if (c.end !== undefined) return BigInt(c.end);
  const w = widthOf(c.kind), draws = c.kind === "fill_normal_f32" ? c.n + (c.n % 2) : c.n;
  return align(BigInt(c.start), w) + BigInt(w * draws);
}

/** The elements the checklist cuts a fill case at. A Float32 normal fill cuts only at pair
 * boundaries: 2, 8, 20 and the largest even element below n. */
export function cutsOf(c: Raw): number[] {
  const at = c.kind === "fill_normal_f32"
    ? [2, 8, 20, c.n - 1 - ((c.n - 1) % 2)]
    : [1, 7, 20, 21, c.n - 1];
  return at.filter((k) => k > 0 && k < c.n);
}

/** True when y passes against the fixture x: bit for bit, or within the case's tolerance. A
 * Float64 exponential rounds each multiply-add twice here and stays within 4 ulps. */
export function agrees(c: Raw, got: ArrayLike<number | bigint>): boolean {
  const want = expected(c);
  if (got.length !== want.length) return false;
  const near = c.tol
    ? (y: number, x: number) => Math.abs(y - x) <= 16 * 2 ** -23 * Math.abs(x) + 1e-6
    : c.kind === "fill_exponential_f64"
    ? (y: number, x: number) => Math.abs(y - x) <= 4 * 2 ** -52 * Math.abs(x)
    : (y: number, x: number) => Object.is(y, x);
  for (let i = 0; i < want.length; i++) {
    const y = got[i], x = want[i];
    if (typeof x === "bigint" ? y !== x : !near(y as number, x)) return false;
  }
  return true;
}

export const sha256 = (chunks: ArrayBufferView[]) => {
  const h = createHash("sha256");
  for (const c of chunks) h.update(new Uint8Array(c.buffer, c.byteOffset, c.byteLength));
  return h.digest("hex");
};

/** 64-bit FNV-1a of the bytes, on 32-bit halves: the prime is 2^40 + 0x1b3. */
export function fnv1a(chunks: ArrayBufferView[]): string {
  let lo = 0x84222325, hi = 0xcbf29ce4;
  for (const c of chunks) {
    const b = new Uint8Array(c.buffer, c.byteOffset, c.byteLength);
    for (let i = 0; i < b.length; i++) {
      lo = (lo ^ b[i]) >>> 0;
      const p = lo * 0x1b3;
      hi = (Math.imul(hi, 0x1b3) + (lo << 8) + Math.floor(p / 2 ** 32)) >>> 0;
      lo = p >>> 0;
    }
  }
  return hi.toString(16).padStart(8, "0") + lo.toString(16).padStart(8, "0");
}
