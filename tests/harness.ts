// One test API for Node, Deno and Bun: node:test and node:assert run on all three.
import { readFile } from "node:fs/promises";
import { deepStrictEqual, throws } from "node:assert/strict";

export { test } from "node:test";

/** Strict deep equality: typed arrays compare element by element, and -0 differs from 0. */
export const assertEquals = (got: unknown, want: unknown, message?: string) =>
  message === undefined ? deepStrictEqual(got, want) : deepStrictEqual(got, want, message);

export const assertThrows = (run: () => unknown, type: new (...args: never[]) => Error) =>
  throws(run, type);

/** A copy of a file in tests/data, so its buffer is exact and aligned. */
export const dump = async (name: string) =>
  new Uint8Array(await readFile(new URL(`./data/${name}`, import.meta.url)));
