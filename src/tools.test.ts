/**
 * The containment rule, which is the one part of a tool that has to be right.
 * Everything else a tool does is its own business; leaving the workspace is not.
 */
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, test } from "vitest";

import { listFilesTool, readFileTool, writeNoteTool } from "./tools.js";

let root = "";

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "agent-tools-"));
  await writeFile(join(root, "hello.txt"), "one\ntwo\n", "utf8");
  await mkdir(join(root, "inner"));
  await writeFile(join(root, "inner", "deep.txt"), "deep\n", "utf8");
});

describe("workspace tools", () => {
  test("reads a file inside the workspace", async () => {
    const fileText = await readFileTool(root).execute({ path: "hello.txt" });
    expect(fileText).toBe("one\ntwo\n");
  });

  test("lists a directory, marking the directories", async () => {
    const listing = await listFilesTool(root).execute({ path: "." });
    expect(listing.split("\n")).toEqual(["hello.txt", "inner/"]);
  });

  test("a path that climbs out is refused by name", async () => {
    await expect(
      readFileTool(root).execute({ path: "../../etc/passwd" }),
    ).rejects.toThrow(/leaves the workspace/);
  });

  test("an absolute path is refused the same way", async () => {
    await expect(
      readFileTool(root).execute({ path: "/etc/passwd" }),
    ).rejects.toThrow(/leaves the workspace/);
  });

  test("a missing path argument is refused before the filesystem is touched", async () => {
    await expect(readFileTool(root).execute({})).rejects.toThrow(
      /non-empty string/,
    );
  });

  test("a directory handed to readFile says which tool to use", async () => {
    await expect(readFileTool(root).execute({ path: "inner" })).rejects.toThrow(
      /listFiles/,
    );
  });

  test("the note tool declares that it needs approval", () => {
    expect(writeNoteTool(root).needsApproval).toBe(true);
  });

  test("the note tool really appends, which is what the gate is guarding", async () => {
    const tool = writeNoteTool(root);
    expect(await tool.execute({ line: "first" })).toContain("appended");
    await tool.execute({ line: "second" });
    const writtenText = await readFile(join(root, "notes.txt"), "utf8");
    expect(writtenText).toBe("first\nsecond\n");
  });

  test("an empty line is refused before the file is touched", async () => {
    await expect(writeNoteTool(root).execute({ line: "" })).rejects.toThrow(
      /non-empty/,
    );
  });
});
