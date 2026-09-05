/**
 * Two tools, and the one safety idea worth taking from the reference agent.
 *
 * That one resolves every path token out of a shell command and refuses what
 * leaves the trust root — a hundred lines, because its `bash` tool can reach
 * anywhere. These two take a path and nothing else, so containment is a
 * `resolve` and a prefix test, and that is the whole of it. Shipping a shell
 * tool would mean shipping the hundred lines too; this package does not.
 */

import { appendFile, readFile, readdir, stat } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { jsonSchema, type JsonSchema } from "modelpact";

import type { Tool } from "./agent.js";

const toSchema = (shape: Record<string, unknown>): JsonSchema => {
  const builtSchema = jsonSchema(shape);
  if (builtSchema === null) throw new Error("not a schema");
  return builtSchema;
};

const PATH_ARG = toSchema({
  type: "object",
  properties: { path: { type: "string" } },
  required: ["path"],
  additionalProperties: false,
});

/** Inside, or refused by name. `..` and an absolute path both land outside and are caught the same way. */
const resolveWithin = (root: string, askedPath: unknown): string => {
  if (typeof askedPath !== "string" || askedPath === "")
    throw new Error("path must be a non-empty string");
  const fullPath = resolve(root, askedPath);
  const relativePath = relative(root, fullPath);
  const isOutsideRoot =
    relativePath.startsWith(`..${sep}`) ||
    relativePath === ".." ||
    relativePath.startsWith(sep);
  if (isOutsideRoot) throw new Error(`path leaves the workspace: ${askedPath}`);
  return fullPath;
};

export const readFileTool = (root: string): Tool => ({
  name: "readFile",
  description: "Read a UTF-8 text file inside the workspace. Args: { path }.",
  parameters: PATH_ARG,
  execute: async (args) => {
    const fullPath = resolveWithin(root, args.path);
    const entryStats = await stat(fullPath);
    if (entryStats.isDirectory())
      throw new Error("that is a directory; use listFiles");
    return readFile(fullPath, "utf8");
  },
});

export const listFilesTool = (root: string): Tool => ({
  name: "listFiles",
  description:
    'List the entries of a directory inside the workspace. Args: { path }, "." for the root.',
  parameters: PATH_ARG,
  execute: async (args) => {
    const fullPath = resolveWithin(root, args.path ?? ".");
    const entries = await readdir(fullPath, { withFileTypes: true });
    if (entries.length === 0) return "(empty)";
    const lines = entries.map((entry) =>
      entry.isDirectory() ? `${entry.name}/` : entry.name,
    );
    return lines.sort().join("\n");
  },
});

/**
 * The one tool here that changes something, and therefore the one behind the
 * gate. It really appends: a tool that asks permission and then does nothing
 * teaches the opposite of what the gate is for, and it lies to the model —
 * asked to write, told "would write", the model reports back that it wrote.
 *
 * The gate is the safety, and containment is the same `within` the readers use,
 * so the worst an approved call can do is add a line to one file in the
 * workspace.
 */
export const writeNoteTool = (root: string): Tool => ({
  name: "writeNote",
  description: "Append a line to notes.txt in the workspace. Args: { line }.",
  parameters: toSchema({
    type: "object",
    properties: { line: { type: "string" } },
    required: ["line"],
    additionalProperties: false,
  }),
  needsApproval: true,
  execute: async (args) => {
    const line = args.line;
    if (typeof line !== "string" || line === "")
      throw new Error("line must be a non-empty string");
    const fullPath = resolveWithin(root, "notes.txt");
    await appendFile(fullPath, `${line}\n`, "utf8");
    return `appended to notes.txt: ${line}`;
  },
});
