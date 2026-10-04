// Write tests/cross.json from the C fixtures of tandem-c and tandem-cuda, so the TypeScript
// tests check the same expected values as those ports. The headers are read as data: each
// `static const ... NAME = { ... };` becomes a JSON value, structs as positional arrays.
// 64-bit integers become decimal strings. Usage, from the repo root with the checkouts beside
// it: deno run --allow-read --allow-write tools/gen_cross.ts
const SOURCES: Record<string, string[]> = {
  "../tandem-c/tests/cross_below.h": ["CROSS_U32", "CROSS_U64"],
  "../tandem-c/tests/cross_fill_below.h": ["CROSS_FILL_U32", "CROSS_FILL_U64"],
  "../tandem-c/tests/cross_normal.h": [
    "CROSS_NORMAL",
    "CROSS_NORMAL_END_POS",
    "CROSS_NORMALF",
    "CROSS_NORMALF_END_POS",
  ],
  "../tandem-cuda/tests/cross_fill_below.h": ["CROSS_FILL_KEY", "CROSS_BELOW32", "CROSS_BELOW64"],
  "../tandem-cuda/tests/cross_fill_normal.h": ["CROSS_NORMAL64", "CROSS_NORMAL32"],
};

function toJson(body: string): unknown {
  const json = body
    .replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, "")
    .replace(/\{/g, "[")
    .replace(/\}/g, "]")
    .replace(/\b0x([0-9a-f]+)u?\b/gi, (_, h) => String(parseInt(h, 16)))
    .replace(/\b(\d+)(?:ull|u)\b/gi, '"$1"')
    .replace(/(\d)f\b/g, "$1")
    .replace(/,\s*\]/g, "]");
  return JSON.parse(json);
}

/** The initializer after `NAME[...] =`, up to the `;` outside braces. */
function initializer(text: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}\\b(?:\\[[^\\]]*\\])*\\s*=\\s*`).exec(text);
  if (!m) return undefined;
  let depth = 0;
  for (let i = m.index + m[0].length; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") depth--;
    else if (text[i] === ";" && depth === 0) return text.slice(m.index + m[0].length, i);
  }
}

const out: Record<string, unknown> = {};
for (const [file, names] of Object.entries(SOURCES)) {
  const text = await Deno.readTextFile(new URL(`../${file}`, import.meta.url));
  for (const name of names) {
    const body = initializer(text, name);
    if (body === undefined) throw new Error(`${name} not found in ${file}`);
    out[name] = toJson(body);
  }
}
await Deno.writeTextFile(
  new URL("../tests/cross.json", import.meta.url),
  JSON.stringify(out) + "\n",
);
