// Normals and exponentials on the CPU with the arithmetic of tandem-c: the Float64 ziggurat
// with the spec's tables, a short polynomial logarithm, Taylor series for cos and sin, and an
// explicit fused multiply-add at every multiply-add. JavaScript has no fma, so `fma64` and
// `fma32` emulate it exactly, which makes the values equal tandem-c bit for bit on any engine.
import { ZIG_K, ZIG_R, ZIG_W, ZIG_Y } from "./zig_tables.ts";

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer);
const F32 = new Float32Array(1);
const I32 = new Uint32Array(F32.buffer);
// Word indices of the low and high half of a double. The engines are little-endian.
const LO = 0, HI = 1;

const SPLIT = 134217729; // 2^27 + 1, Veltkamp's splitter

/** w * b + c rounded once, where w = wh + wl is already split by Veltkamp's method. Dekker's
 * product gives p + e = w * b exactly and the 2Sum of c and p gives s + t, so the total is
 * s + (t + e). When that inner sum is exact the outer rounding is the fused result. Otherwise
 * only a tie of the outer sum can go wrong, and the sign of the lost part settles it. Arguments
 * stay clear of overflow and of the subnormal range, which holds for the polynomials below. */
function fmaw(w: number, wh: number, wl: number, b: number, c: number): number {
  const t = SPLIT * b, bh = t - (t - b), bl = b - bh;
  const p = w * b;
  const e = ((wh * bh - p) + wh * bl + wl * bh) + wl * bl;
  const s = c + p, bb = s - c;
  const t1 = (c - (s - bb)) + (p - bb);
  const v = t1 + e, bv = v - t1, ev = (t1 - (v - bv)) + (e - bv);
  const z = s + v;
  if (ev !== 0) {
    // The exact total is s + v + ev, with ev far below the spacing of s + v, so z is wrong only
    // when s + v is exactly halfway between two doubles and ev decides the tie.
    const bz = z - s, r = (s - (z - bz)) + (v - bz), r2 = r + r;
    if (r !== 0 && (z + r2) - z === r2 && (ev > 0) === (r > 0)) return z + r2;
  }
  return z;
}

/** a * b + c rounded once, as a fused multiply-add. */
export function fma64(a: number, b: number, c: number): number {
  const t = SPLIT * a, ah = t - (t - a);
  return fmaw(a, ah, a - ah, b, c);
}

/** The polynomial c[0] w^(n-1) + ... + c[n-1] by fused Horner steps, as the nested fma calls of
 * tandem-c. The split of w is shared by all steps. Each step adds the coefficient to a product
 * that is smaller in magnitude, which holds for the logarithm's polynomial below, so the sum
 * needs only Fast2Sum, and the lost part of the second sum is read only on a tie. */
export function horner64(w: number, c: Float64Array): number {
  const t0 = SPLIT * w, wh = t0 - (t0 - w), wl = w - wh;
  let acc = c[0];
  for (let k = 1; k < c.length; k++) {
    const cc = c[k], t = SPLIT * acc, bh = t - (t - acc), bl = acc - bh;
    const p = w * acc, e = ((wh * bh - p) + wh * bl + wl * bh) + wl * bl;
    const s = cc + p, t1 = p - (s - cc);
    // When t1 is not 0, |s| exceeds |v| by far, so the Fast2Sum below is exact. When it is 0,
    // v = e exactly and z is already the fused result.
    const v = t1 + e, z = s + v, r = v - (z - s), r2 = r + r;
    acc = z;
    if (r !== 0 && (z + r2) - z === r2) {
      const bv = v - t1, ev = (t1 - (v - bv)) + (e - bv);
      if (ev !== 0 && (ev > 0) === (r > 0)) acc = z + r2;
    }
  }
  return acc;
}

/** The Float32 fused multiply-add on Float32 values held in doubles. The product is exact in a
 * double, so the sum rounds twice, to double and to Float32. That is wrong only when the double
 * sum is exactly halfway between two Float32 values, which its low mantissa bits show, and its
 * own lost part then settles the tie. */
export function fma32(a: number, b: number, c: number): number {
  const p = a * b, s = p + c;
  F64[0] = s;
  const lo = U32[LO];
  if ((lo & 0x1fffffff) === 0x10000000) {
    const bb = s - c, e = (c - (s - bb)) + (p - bb);
    if (e !== 0) {
      U32[LO] = (e > 0) === (s > 0) ? lo + 1 : lo - 1;
      return Math.fround(F64[0]);
    }
  }
  return Math.fround(s);
}

export const LOG_POLY = Float64Array.of(
  0.08312363319426472,
  0.09070001083303751,
  0.11111433317907482,
  0.14285712049336274,
  0.2000000000566491,
  0.33333333333331017,
  1.0,
);
// ln 2 split so that nk * LN2_HI is exact.
const LN2_HI = 1.3862943607382476, LN2_LO = 3.816429394731813e-10;
const LN2_HI_H = SPLIT * LN2_HI - (SPLIT * LN2_HI - LN2_HI), LN2_HI_L = LN2_HI - LN2_HI_H;
const LN2_LO_H = SPLIT * LN2_LO - (SPLIT * LN2_LO - LN2_LO), LN2_LO_L = LN2_LO - LN2_LO_H;

/** -2 ln x for x in (0, 1]: x = mant 2^k with mant in [sqrt(1/2), sqrt(2)) from the bits, then
 * 2 k ln 2 - 4 s p(s^2) with s = (mant - 1) / (mant + 1). */
export function neg2Log64(x: number): number {
  F64[0] = x;
  const ix = U32[HI] + 0x00095f62;
  const nk = 1023 - (ix >>> 20);
  U32[HI] = (ix & 0xfffff) + 0x3fe6a09e;
  const mant = F64[0];
  const s = (mant - 1) / (mant + 1), p = horner64(s * s, LOG_POLY);
  return fmaw(LN2_LO, LN2_LO_H, LN2_LO_L, nk, fmaw(LN2_HI, LN2_HI_H, LN2_HI_L, nk, (s * -4.0) * p));
}

const fr = Math.fround;
const LOG_POLY32 = Float32Array.of(0.14275366, 0.20000061, 0.33333334, 1);
const L32_HI = fr(1.38629150390625), L32_LO = fr(2.857213530660374e-06);

/** The Float32 polynomial c[0] w^(n-1) + ... + c[n-1] by fused steps, on Float32 values. */
export function horner32(w: number, c: Float32Array): number {
  let acc = c[0];
  for (let k = 1; k < c.length; k++) acc = fma32(w, acc, c[k]);
  return acc;
}

export function neg2Log32(x: number): number {
  F32[0] = x;
  const ix = I32[0] + 0x004afb0d;
  const nk = 127 - (ix >>> 23);
  I32[0] = (ix & 0x7fffff) + 0x3f3504f3;
  const mant = F32[0];
  const s = fr(fr(mant - 1) / fr(mant + 1)), p = horner32(fr(s * s), LOG_POLY32);
  return fma32(nk, L32_LO, fma32(nk, L32_HI, fr(fr(s * -4) * p)));
}

// The widths W[i] for a clear sign bit and -W[i] at 1024 + i for a set one, so the table index
// is the low 11 bits of a draw and the sign needs no branch.
const ZW = new Float64Array(2048);
ZW.set(ZIG_W);
for (let i = 0; i < 1024; i++) ZW[1024 + i] = -ZIG_W[i];

/** The fast path of the Float64 ziggurat (Appendix A) for the 64-bit draw with words lo and hi:
 * the normal, or NaN when the draw misses the inner rectangle of its layer. */
export function zigFast(lo: number, hi: number): number {
  const ra = hi * 2097152 + (lo >>> 11);
  return ra < ZIG_K[lo & 1023] ? ra * ZW[lo & 2047] : NaN;
}

/** The 64-bit draws of a generator in sequence, as the words of the last one. */
export interface Draws64 {
  lo: number;
  hi: number;
  next(): void;
}

const uniform = (f: Draws64) => {
  f.next();
  return (f.hi * 2097152 + (f.lo >>> 11)) * 2 ** -53;
};

/** The slow path of the ziggurat from a missed draw, on the draws of its fallback generator.
 * ln y is -0.5 neg2Log64(y), and every other operation rounds once, as in tandem-c. */
export function zigSlow(lo: number, hi: number, f: Draws64): number {
  for (;;) {
    const i = lo & 1023, ra = hi * 2097152 + (lo >>> 11), x = ra * ZW[lo & 2047];
    if (ra < ZIG_K[i]) return x;
    if (i === 0) {
      // The tail beyond R, by Marsaglia's method.
      let a: number, b: number;
      do {
        a = 0.5 * neg2Log64(1 - uniform(f)) / ZIG_R;
        b = 0.5 * neg2Log64(1 - uniform(f));
      } while (b + b < a * a);
      return lo & 1024 ? -(ZIG_R + a) : ZIG_R + a;
    }
    const y = ZIG_Y[i] + uniform(f) * (ZIG_Y[i + 1] - ZIG_Y[i]);
    if (-0.5 * neg2Log64(y) < -0.5 * (x * x)) return x;
    f.next();
    lo = f.lo;
    hi = f.hi;
  }
}

export const SIN_POLY32 = Float32Array.of(2.72499e-06, -0.00019840087, 0.008333332, -0.16666667, 1);
export const COS_POLY32 = Float32Array.of(2.4463761e-05, -0.0013887589, 0.04166665, -0.5, 1);
const TWO_PI_HI = fr(6.2831855), TWO_PI_LO = fr(-1.7484555e-7);

export function normalPairs32(z: Float32Array, m: number): void {
  for (let j = 0; j < m; j++) {
    const a = z[2 * j], b = z[2 * j + 1];
    const r = fr(Math.sqrt(neg2Log32(fr(1 - a))));
    const q = fr(b * 4 + 0.5) | 0;
    const f = fr(b - q * 0.25);
    const th = fma32(f, TWO_PI_LO, fr(f * TWO_PI_HI)), w = fr(th * th);
    const sn = fr(th * horner32(w, SIN_POLY32)), cs = horner32(w, COS_POLY32);
    let x: number, y: number;
    switch (q & 3) {
      case 0:
        x = cs;
        y = sn;
        break;
      case 1:
        x = -sn;
        y = cs;
        break;
      case 2:
        x = -cs;
        y = -sn;
        break;
      default:
        x = sn;
        y = -cs;
    }
    z[2 * j] = fr(r * x);
    z[2 * j + 1] = fr(r * y);
  }
}

/** In place, z[j] = -ln(1 - z[j]) for uniforms z[j]. Halving -2 ln is exact. */
export function exponential64(z: Float64Array, m: number): void {
  for (let j = 0; j < m; j++) z[j] = 0.5 * neg2Log64(1 - z[j]);
}

export function exponential32(z: Float32Array, m: number): void {
  for (let j = 0; j < m; j++) z[j] = 0.5 * neg2Log32(fr(1 - z[j]));
}
