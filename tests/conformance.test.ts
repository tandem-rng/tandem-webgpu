// The CPU path against the conformance files of the specification, one test per section of
// its conformance/CHECKLIST.md at tandem-spec b31af72.
import { existsSync } from "node:fs";
import { block, ChoiceTable, seed, Tandem } from "../src/mod.ts";
import {
  agrees,
  allCases,
  byName,
  cases,
  cutsOf,
  endOf,
  expected,
  f64,
  fnv1a,
  hashes,
  keyOf,
  sha256,
} from "./fixtures.ts";
import { assertEquals, assertThrows, dump, test } from "./harness.ts";

type Case = ReturnType<typeof byName>;
type Values = Uint32Array | BigUint64Array | Float32Array | Float64Array;

const generator = (c: Case) => new Tandem(keyOf(c.key), { position: BigInt(c.start), K: c.K });
const tables = new Map<Case, ChoiceTable>();
const tableOf = (c: Case) => {
  if (!tables.has(c)) tables.set(c, new ChoiceTable(c.weights!.map(f64)));
  return tables.get(c)!;
};
const rangeOf = (c: Case) => BigInt(`0x${c.range}`);

/** The operation of the case's kind for n elements on g: n scalar draws for `below_*`. */
function run(c: Case, g: Tandem, n: number): Values {
  switch (c.kind) {
    case "below_u32":
      return Uint32Array.from({ length: n }, () => g.nextU32Below(Number(rangeOf(c))));
    case "below_u64":
      return BigUint64Array.from({ length: n }, () => g.nextU64Below(rangeOf(c)));
    case "fill_below_u32":
      return g.fillU32Below(n, Number(rangeOf(c)));
    case "fill_below_u64":
      return g.fillU64Below(n, rangeOf(c));
    case "fill_normal_f64":
      return g.fillNormalF64(n);
    case "fill_normal_f32":
      return g.fillNormalF32(n);
    case "fill_exponential_f64":
      return g.fillExponentialF64(n);
    case "fill_exponential_f32":
      return g.fillExponentialF32(n);
    case "fill_choice":
      return g.fillChoice(n, tableOf(c));
  }
  throw new Error(`unknown kind ${c.kind}`);
}

/** The scalar draw of the kinds whose scalar equals element 0 of a fill. */
const SCALAR: Record<string, (g: Tandem, c: Case) => number> = {
  fill_normal_f64: (g) => g.nextNormalF64(),
  fill_exponential_f64: (g) => g.nextExponentialF64(),
  fill_exponential_f32: (g) => g.nextExponentialF32(),
  fill_choice: (g, c) => g.nextChoice(tableOf(c)),
};

test("every case: values and end position, scalar draws equal to the fills", () => {
  for (const c of allCases()) {
    const g = generator(c), got = run(c, g, c.n);
    assertEquals(agrees(c, got), true, `${c.id} values`);
    assertEquals(g.position, endOf(c), `${c.id} end`);
    const scalar = SCALAR[c.kind];
    if (!scalar || c.n === 0) continue;
    const one = generator(c), each = Array.from({ length: c.n }, () => scalar(one, c));
    assertEquals(each, Array.from(got as ArrayLike<number>), `${c.id} scalar draws`);
    assertEquals(one.position, g.position, `${c.id} scalar end`);
  }
});

test("fallback by global draw index: a later start continues the same fill", () => {
  for (
    const [later, earlier] of [
      ["CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"],
      ["CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"],
      ["CROSS_NORMAL[1]", "CROSS_NORMAL[0]"],
      ["CROSS_CHOICE[1]", "CROSS_CHOICE[0]"],
    ]
  ) {
    const a = byName(later), b = byName(earlier);
    const got = run(a, generator(a), a.n), whole = run(b, generator(b), b.n);
    assertEquals(got.subarray(0, a.n - 1), whole.subarray(1), `${later} after ${earlier}`);
  }
});

test("width from range: a 64-bit result of range 1000 takes 32-bit draws", () => {
  const narrow = byName("CROSS_BELOW32[3]"), wide = byName("CROSS_BELOW64[3]");
  assertEquals(keyOf(narrow.key), seed(42n));
  const g = generator(narrow);
  assertEquals(
    g.fillBelow(64, 1000n),
    BigUint64Array.from(expected(narrow) as Uint32Array, BigInt),
  );
  assertEquals(g.position, 32n * 64n);
  assertEquals(generator(wide).fillU64Below(64, 1000n), expected(wide));
  // Range 0 gives 0 and consumes one draw of the width the range selects.
  const h = Tandem.seed(5n);
  assertEquals([h.nextBelow(0), h.position], [0, 32n]);
  assertEquals([h.nextBelow(0n), h.position], [0n, 64n]);
  assertEquals([h.nextU64Below(0n), h.position], [0n, 128n]);
  assertEquals([h.fillU32Below(3, 0), h.position], [new Uint32Array(3), 224n]);
});

test("odd n and the pair rule of Float32 Box-Muller", () => {
  const n0 = byName("CROSS_NORMAL32[0]"), g = generator(n0);
  g.fillNormalF32(33);
  assertEquals(g.position, 1088n);
  // A start one pair later shifts the output by one pair.
  const n2 = byName("CROSS_NORMAL32[2]");
  assertEquals(run(n2, generator(n2), 31), run(n0, generator(n0), 33).subarray(2));
  // The pairs from start 1 are the pairs from 32.
  const f = byName("CROSS_NORMALF"), n1 = byName("CROSS_NORMAL32[1]");
  assertEquals(run(f, generator(f), f.n).subarray(0, 33), run(n1, generator(n1), 33));
  // A scalar Float32 normal is the cosine half and consumes both draws.
  const one = Tandem.seed(8n);
  assertEquals(one.nextNormalF32(), Tandem.seed(8n).fillNormalF32(2)[0]);
  assertEquals(one.position, 64n);
});

test("weighted choice: tables, scalar draws, m = 1 and rejected weights", () => {
  const withTable = cases("choice").filter((c) => c.cut);
  assertEquals(withTable.length, 5);
  for (const c of withTable) {
    const t = tableOf(c);
    assertEquals(t.cut, BigUint64Array.from(c.cut!, (h) => BigInt(`0x${h}`)), c.id);
    assertEquals(t.alias, Uint32Array.from(c.alias!, (h) => parseInt(h, 16)), c.id);
  }
  for (const c of cases("choice")) {
    assertEquals(tableOf(c).capacity, BigInt(`0x${c.capacity}`), c.id);
  }
  const g = Tandem.seed(3n, 8), t = new ChoiceTable([1, 2, 3, 4]);
  assertEquals(g.nextChoice(t), new Tandem(seed(3n), { K: 8 }).fillChoice(1, t)[0]);
  assertEquals(g.position, 64n);
  const single = new ChoiceTable([7]);
  assertEquals([g.nextChoice(single), g.position], [0, 128n]);
  for (const w of [[], [1, -1], [1, NaN], [1, Infinity], [0, -0]]) {
    assertThrows(() => new ChoiceTable(w), RangeError);
  }
});

test("weighted choice: 1e6 draws follow the weights by chi-square", () => {
  // A zero weight never appears. Nine positive weights leave 8 degrees of freedom: the 0.0005
  // and 0.9995 quantiles are 0.71 and 27.87.
  const w = [0.5, 3, 0, 1, 7, 2.25, 0.1, 4, 1, 6], n = 1_000_000;
  const x = Tandem.seed(2028n).fillChoice(n, new ChoiceTable(w)), counts = new Array(10).fill(0);
  for (const i of x) counts[i]++;
  assertEquals(counts[2], 0);
  const total = w.reduce((a, b) => a + b);
  let chi2 = 0;
  for (let i = 0; i < w.length; i++) {
    if (w[i] > 0) chi2 += (counts[i] - n * w[i] / total) ** 2 / (n * w[i] / total);
  }
  assertEquals(0.71 < chi2 && chi2 < 27.87, true, `chi2 = ${chi2}`);
});

test("cut fill: pieces filled in order on one generator equal the whole fill", () => {
  for (const c of allCases().filter((c) => c.kind.startsWith("fill_") && c.n > 0)) {
    const whole = run(c, generator(c), c.n);
    for (const k of cutsOf(c)) {
      const g = generator(c), head = run(c, g, k), tail = run(c, g, c.n - k);
      assertEquals(head, whole.subarray(0, k), `${c.id} head ${k}`);
      assertEquals(tail, whole.subarray(k), `${c.id} tail ${k}`);
      assertEquals(g.position, endOf(c), `${c.id} cut ${k}`);
    }
  }
});

// Instead of the checklist items on UInt128, Float16, Char and complex draws: this port has no
// draw of these types, so it hashes the streams of the types it fills and skips the others,
// and the complex draw across a block does not apply.
const STREAM_FILLS: Record<string, (g: Tandem, n: number) => ArrayBufferView> = {
  UInt8: (g, n) => g.fillU8(n),
  UInt32: (g, n) => g.fillU32(n),
  UInt64: (g, n) => g.fillU64(n),
  Float32: (g, n) => g.fillF32(n),
  Float64: (g, n) => g.fillF64(n),
  Bool: (g, n) => g.fillBool(n),
};

test("stream hashes: the fills and the dumps in tests/data", async () => {
  let filled = 0, dumps = 0;
  for (const s of hashes.streams) {
    const fill = STREAM_FILLS[s.type];
    if (fill) {
      const g = new Tandem(keyOf(s.key), { position: BigInt(s.start), K: s.K });
      assertEquals(sha256([fill(g, s.n)]), s.sha256, s.file);
      filled++;
    }
    const name = s.file.replace("tests/data/", "");
    if (existsSync(new URL(`./data/${name}`, import.meta.url))) {
      assertEquals(sha256([await dump(name)]), s.sha256, name);
      dumps++;
    }
  }
  assertEquals([filled, dumps], [7, 6]);
});

// One test per dump, so each stays under Bun's 5 s default timeout.
for (const d of hashes.dumps) {
  const exact = !d.draws.some((x) => x.kind === "fill_exponential_f64");
  // The Float64 exponentials round twice and differ from the dump. cpu.test.ts checks its
  // hash on the exact form.
  if (!exact) continue;
  test(`dump hash: ${d.id} from ${d.starts.length} start(s)`, () => {
    const out: ArrayBufferView[] = [];
    let g!: Tandem;
    for (const start of d.starts) {
      g = new Tandem(keyOf(d.key), { position: BigInt(start), K: d.K });
      for (const { kind, n } of d.draws) {
        out.push(kind === "fill_normal_f64" ? g.fillNormalF64(n) : g.fillNormalF32(n));
      }
    }
    assertEquals(fnv1a(out), d.fnv1a);
    if (d.sha256) assertEquals(sha256(out), d.sha256);
    if (d.end) assertEquals(g.position, BigInt(d.end));
  });
}

test("2^63 position bounds: starts, and a UInt64 draw at 2^63 - 1", () => {
  const key = seed(4n), top = (1n << 63n) - 1n;
  for (const position of [1n << 63n, (1n << 64n) - 1n, -1n]) {
    assertThrows(() => new Tandem(key, { position }), RangeError);
  }
  const g = new Tandem(key, { position: top });
  assertEquals(g.position, top);
  // Bit 2^63 is block 2^56: row 2^53, lane 0.
  const b = block(key, 8n * ((1n << 53n) / 32n), Number((1n << 53n) % 32n));
  assertEquals(g.nextU64(), BigInt(b[0]) | (BigInt(b[1]) << 32n));
  assertEquals(g.position, (1n << 63n) + 64n);
  // Instead of the checklist item on a fill that reaches 2^64: starts are rejected at or above
  // 2^63, and a fill count is a number below 2^53, so a fill can not reach 2^64 bits.
});
