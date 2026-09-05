/**
 * The real thing, when it is here: a daemon on 11434 with a model big enough
 * to hold a tool-calling shape. Skips loudly otherwise.
 *
 * What it proves is the one thing a stubbed brain cannot: that a real model,
 * handed real tools, calls one and answers from what came back. The other
 * branch of the loop — a brain that refuses schemas, so the run goes through
 * prose — is reached by a stub in `agent.test.ts`; reaching it live wants a
 * transport that refuses, and that transport lives in
 * `modelpact-orchestrator`.
 */

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { makeOllamaProvider } from "modelpact-providers";

import { runAgent, type AgentEvent } from "./agent.js";
import { makeSessionBrain } from "./brain.js";
import { listFilesTool, readFileTool } from "./tools.js";

/** Big enough to hold a tool-calling shape; a 350m model fills the fields but not the intent. */
const AGENT_MODEL = "qwen3:14b";

const isOllamaHere = await (async () => {
  try {
    const response = await fetch("http://127.0.0.1:11434/api/tags", {
      signal: AbortSignal.timeout(2_000),
    });
    return response.ok;
  } catch {
    return false;
  }
})();

describe.skipIf(!isOllamaHere)("live: the agent loop on a real model", () => {
  test("it calls a tool, reads the result, and answers from it", async () => {
    const root = await mkdtemp(join(tmpdir(), "agent-live-"));
    await writeFile(
      join(root, "colour.txt"),
      "The colour on file is teal.\n",
      "utf8",
    );

    const access = await makeOllamaProvider({
      model: AGENT_MODEL,
      contextWindow: 8192,
    }).access();
    if (access.kind !== "ready")
      throw new Error(`expected ready, got ${access.kind}`);
    const sessionResult = await access.open();
    if (!sessionResult.ok)
      throw new Error(`open refused: ${sessionResult.error.kind}`);

    const events: AgentEvent[] = [];
    const runResult = await runAgent(
      {
        brain: makeSessionBrain(sessionResult.value),
        tools: [listFilesTool(root), readFileTool(root)],
        maxSteps: 6,
        onEvent: (event) => events.push(event),
      },
      "Read colour.txt in the workspace and tell me the colour it names. Use the tools.",
    );

    expect(runResult.ok).toBe(true);
    if (!runResult.ok) return;
    // The answer can only contain the word if the tool actually ran and its
    // output came back into the conversation.
    expect(runResult.value.text.toLowerCase()).toContain("teal");
    expect(runResult.value.constrained).toBe(true);
    expect(
      events.filter((event) => event.kind === "tool").length,
    ).toBeGreaterThan(0);
    sessionResult.value.close();
  }, 300_000);
});
