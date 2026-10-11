// Normals and exponentials on the CPU with the arithmetic of tandem-c: the Float64 ziggurat
// with the spec's tables, a short polynomial logarithm, Taylor series for cos and sin, and a
// multiply-add at every place tandem-c fuses one. JavaScript has no fma. The slow path of the
// Float64 ziggurat emulates it exactly, `fma64`, so the Float64 normals equal tandem-c bit for
// bit on any engine. The exponentials fuse exactly too, `fmaw` and `fmaf`, and equal tandem-c
// bit for bit. The Float32 normals round the plain sum, which is faster and agrees with
// tandem-c within the ulp tolerance of Appendix A.
import { expBlock, expConsts, kernel } from "./stream.ts";
import { ZIG_K, ZIG_R, ZIG_W, ZIG_Y } from "./zig_tables.ts";

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer);
const F32 = new Float32Array(1);
const I32 = new Uint32Array(F32.buffer);
// The word index of the high half of a double. The engines are little-endian.
const HI = 1;

const SPLIT = 134217729; // 2^27 + 1, Veltkamp's splitter

/** c + p + e rounded once, for p + e an exact product. The 2Sum of c and p gives s + t, so the
 * total is s + (t + e). When that inner sum is exact the outer rounding is the fused result.
 * Otherwise only a tie of the outer sum can go wrong, and the sign of the lost part settles it.
 * Arguments stay clear of overflow and of the subnormal range, which holds for the uses below. */
function sumFused(c: number, p: number, e: number): number {
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

/** a * b + c rounded once, as a fused multiply-add: Dekker's product, then the fused sum. */
export function fma64(a: number, b: number, c: number): number {
  const t = SPLIT * a, ah = t - (t - a), al = a - ah;
  const u = SPLIT * b, bh = u - (u - b), bl = b - bh;
  const p = a * b;
  return sumFused(c, p, ((ah * bh - p) + ah * bl + al * bh) + al * bl);
}

/** The spacing of the doubles in the binade of x, for normal positive x. */
function ulp(x: number): number {
  F64[0] = x;
  return 2 ** ((U32[HI] >>> 20) - 1075);
}

/** For the Horner steps of `horner64` on positive coefficients c and w in [0, wmax]: the bound on
 * the rounding error of the plain sum below which the plain sum is the fused result. The step
 * adds the coefficient to a product p that is at most wmax times the largest accumulator, so
 * |p - round(p)| is at most half the spacing of p's binade, and the fused sum is round(s + t + e)
 * for s + t the exact plain sum. When |t| plus that bound stays below half the spacing of s, the
 * fused sum is s. The spacing is the coefficient's: s is at least the coefficient, a power of two
 * above it lies in a higher binade and keeps at least that spacing below, and when s equals the
 * coefficient the lost part t + e is not negative, so the spacing above applies. A zero bound
 * makes the step take the full path. */
export function hornerLimits(c: Float64Array, wmax: number): Float64Array {
  const lim = new Float64Array(c.length);
  let accMax = c[0];
  for (let k = 1; k < c.length; k++) {
    const pmax = wmax * accMax * (1 + 2 ** -40);
    lim[k] = Math.max(0, ulp(c[k]) / 2 - ulp(pmax) / 2);
    accMax = c[k] + pmax;
  }
  return lim;
}

/** The polynomial c[0] w^(n-1) + ... + c[n-1] by fused Horner steps, as the nested fma calls of
 * tandem-c, for lim = hornerLimits(c, wmax) and w in [0, wmax]. A step whose plain sum is proven
 * to be the fused result by its Fast2Sum error, nearly all of them, stops there. The others
 * take Dekker's product. Each step adds the coefficient to a product that is smaller in
 * magnitude, so the sum needs only Fast2Sum, and the lost part of the second sum is read only on
 * a tie. */
export function horner64(w: number, c: Float64Array, lim: Float64Array): number {
  let acc = c[0];
  for (let k = 1; k < c.length; k++) {
    const cc = c[k], p = w * acc, s = cc + p, t1 = p - (s - cc);
    acc = Math.abs(t1) < lim[k] ? s : hornerStep(w, acc, p, s, t1);
  }
  return acc;
}

/** The fused Horner step c + w * acc for the plain sum s with Fast2Sum error t1 and the rounded
 * product p: Dekker's product gives the product's error e, and the step is round(s + t1 + e). */
function hornerStep(w: number, acc: number, p: number, s: number, t1: number): number {
  const t0 = SPLIT * w, wh = t0 - (t0 - w), wl = w - wh;
  const t = SPLIT * acc, bh = t - (t - acc), bl = acc - bh;
  const e = ((wh * bh - p) + wh * bl + wl * bh) + wl * bl;
  // When t1 is not 0, |s| exceeds |v| by far, so the Fast2Sum below is exact. When it is 0,
  // v = e exactly and z is already the fused result.
  const v = t1 + e, z = s + v, r = v - (z - s), r2 = r + r;
  if (r !== 0 && (z + r2) - z === r2) {
    const bv = v - t1, ev = (t1 - (v - bv)) + (e - bv);
    if (ev !== 0 && (ev > 0) === (r > 0)) return z + r2;
  }
  return z;
}

/** The Float32 multiply-add of Float32 values held in doubles. The product is exact, so this
 * is the fused result unless the double sum lies exactly halfway between two Float32 values. */
const fma32 = (a: number, b: number, c: number) => Math.fround(a * b + c);

export const LOG_POLY = Float64Array.of(
  0.08312363319426472,
  0.09070001083303751,
  0.11111433317907482,
  0.14285712049336274,
  0.2000000000566491,
  0.33333333333331017,
  1.0,
);
// 2 ln 2 split so that nk * LN2_HI is exact for the 11-bit exponents nk, and LN2_LO split by
// Veltkamp so that nk times either half is exact.
export const LN2_HI = 1.3862943607382476, LN2_LO = 3.816429394731813e-10;
export const LN2_LO_H = SPLIT * LN2_LO - (SPLIT * LN2_LO - LN2_LO), LN2_LO_L = LN2_LO - LN2_LO_H;

/** The double with the given high and low words. */
function fromWords(hi: number, lo: number): number {
  U32[HI] = hi;
  U32[1 - HI] = lo;
  return F64[0];
}

/** The largest s^2 of `neg2Log64`: s and s^2 are monotone in |mant - 1| through rounding, so the
 * two ends of the mantissa range, high word 0x3fe6a09e to 0x3ff6a09d, give it. */
export const LOG_WMAX = Math.max(
  ...[fromWords(0x3fe6a09e, 0), fromWords(0x3ff6a09d, 0xffffffff)].map((m) =>
    ((m - 1) / (m + 1)) ** 2
  ),
);
export const LOG_LIM = hornerLimits(LOG_POLY, LOG_WMAX);

/** -2 ln x for x in (0, 1]: x = mant 2^k with mant in [sqrt(1/2), sqrt(2)) from the bits, then
 * 2 k ln 2 - 4 s p(s^2) with s = (mant - 1) / (mant + 1). Every multiply-add rounds once as in
 * tandem-c, so the Float64 exponentials and the ziggurat slow path equal tandem-c bit for bit.
 * The product nk * LN2_HI is exact, so its fused add is the plain sum. The last step adds
 * q = nk * LN2_LO, below 2^-21, to y, which is 0 for nk = 0 and else at least 0.69 in magnitude,
 * so the product's error is below 2^-74 and the plain sum is the fused one when its Fast2Sum
 * error stays under LAST_LIM |y|, as in `hornerLimits`. Otherwise the two products of nk with
 * the halves of LN2_LO are exact, so their difference from q is its error without a split. */
export function neg2Log64(x: number): number {
  F64[0] = x;
  const ix = U32[HI] + 0x00095f62;
  const nk = 1023 - (ix >>> 20);
  U32[HI] = (ix & 0xfffff) + 0x3fe6a09e;
  const mant = F64[0];
  const s = (mant - 1) / (mant + 1), w = s * s;
  const y = nk * LN2_HI + (s * -4.0) * horner64(w, LOG_POLY, LOG_LIM);
  const q = nk * LN2_LO, z = y + q, t1 = q - (z - y);
  if (Math.abs(t1) < LAST_LIM * Math.abs(y)) return z;
  return sumFused(y, q, (nk * LN2_LO_H - q) + nk * LN2_LO_L);
}

// Half the spacing of y is above |y| 2^-54, and 2^-74 is below |y| 2^-73 for |y| at least 1/2.
const LAST_LIM = 2 ** -54 - 2 ** -73;

// The table that exp64 of stream.wat reads, in its layout: the polynomial, its limits and the
// constants of neg2Log64.
if (expConsts) {
  expConsts.set(LOG_POLY, 0);
  expConsts.set(LOG_LIM, 8);
  expConsts.set([SPLIT, LN2_HI, LN2_LO, LN2_LO_H, LN2_LO_L, LAST_LIM], 16);
}

const fr = Math.fround;
const LOG_POLY32 = Float32Array.of(0.14275366, 0.20000061, 0.33333334, 1);
const L32_HI = fr(1.38629150390625), L32_LO = fr(2.857213530660374e-06);

/** The Float32 polynomial c[0] w^(n-1) + ... + c[n-1] by Horner steps, on Float32 values. */
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

/** The Float32 multiply-add of Float32 values held in doubles, rounded once. The product is
 * exact, and t is the error of the double sum s, so fround(s) is the fused result unless s lies
 * exactly halfway between two Float32 values. Then the sign of t picks the side. Arguments stay
 * in the normal range, which holds for the logarithm below. */
function fmaf(a: number, b: number, c: number): number {
  const p = a * b, s = p + c, bb = s - p, t = (p - (s - bb)) + (c - bb);
  const r = fr(s);
  if (t === 0 || r === s) return r;
  const h = s - r, other = r + 2 * h;
  return fr(other) === other && (t > 0) === (h > 0) ? other : r;
}

const LOG_Q32 = Float32Array.of(0.0023109776, 0.012496489, 0.08333336);
const LN2_HI32 = fr(0.693145751953125), LN2_LO32 = fr(1.428606765330187e-06);

/** -ln x for x in (0, 1], tandem-c's neg_log_f32, within 0.58 ulp: the Float32 exponential. The
 * leading term u = (2 - 2m) / (m + 1) is carried as uh + r / d, with m + 1 = d + dl exactly and
 * r the residual of uh, and nk ln2_hi + uh is split exactly by fast two-sum. Division, products
 * and sums of Float32 values in double and one fround round exactly once. */
export function negLog32(x: number): number {
  F32[0] = x;
  const ix = I32[0] + 0x004afb0d;
  const nk = 127 - (ix >>> 23);
  I32[0] = (ix & 0x7fffff) + 0x3f3504f3;
  const m = F32[0];
  const num = fmaf(m, -2, 2), d = fr(m + 1), dl = fr(m - fr(d - 1));
  const rcp = fr(1 / d), uh = fr(num * rcp);
  const r = fmaf(-uh, dl, fmaf(-uh, d, num)), v = fr(uh * uh);
  const q = fmaf(v, fmaf(v, LOG_Q32[0], LOG_Q32[1]), LOG_Q32[2]);
  const a = fr(nk * LN2_HI32), hi = fr(a + uh), e = fr(uh - fr(hi - a));
  return fr(hi + fmaf(fr(uh * v), q, fmaf(r, rcp, fmaf(nk, LN2_LO32, e))));
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

/** In place, z[j] = -ln(1 - z[j]) for uniforms z[j]. Halving -2 ln is exact. The WebAssembly
 * kernel maps pairs through its block, with the same arithmetic two lanes at a time, and the
 * JavaScript form takes the odd element and runs everywhere else. */
export function exponential64(z: Float64Array, m: number): void {
  let j = 0;
  const k = kernel();
  if (k && expBlock) {
    for (
      let n = Math.min(m, expBlock.length) & ~1;
      n > 1;
      n = Math.min(m - j, expBlock.length) & ~1
    ) {
      expBlock.set(z.subarray(j, j + n));
      k.exp64(n);
      z.set(expBlock.subarray(0, n), j);
      j += n;
    }
  }
  for (; j < m; j++) z[j] = 0.5 * neg2Log64(1 - z[j]);
}

export function exponential32(z: Float32Array, m: number): void {
  for (let j = 0; j < m; j++) z[j] = negLog32(fr(1 - z[j]));
}
