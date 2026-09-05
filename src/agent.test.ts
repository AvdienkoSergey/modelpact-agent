/**
 * The loop, on a brain made of strings.
 *
 * A scripted brain rather than a model, because what is under test is the
 * control flow: when a tool runs, what reaches the model after it, what a
 * denial and a throw look like from the inside, and where the loop stops.
 * `live.test.ts` runs the same loop on a real model.
 */
import { describe, expect, test } from "vitest";
import { makeMockProvider, type AiFailure, type Result } from "modelpact";

import { runAgent, type AgentEvent, type Tool } from "./agent.js";
import { makeSessionBrain, type Brain } from "./brain.js";

const makeStepReply = (fields: Partial<Record<string, unknown>>): string =>
  JSON.stringify({
    reason: "because",
    action: "answer",
    tool: "",
    args: {},
    answer: "",
    ...fields,
  });

const makeToolCallReply = (
  tool: string,
  args: Record<string, unknown> = {},
): string => makeStepReply({ action: "tool", tool, args });

const makeAnswerReply = (text: string): string =>
  makeStepReply({ action: "answer", answer: text });

interface ScriptedBrain {
  readonly brain: Brain;
  readonly recordedAsks: { input: string; isConstrained: boolean }[];
}

/** Replies in order; runs out into a final answer so a runaway test still ends. */
const makeScriptedBrain = (
  replies: readonly string[],
  refusesSchema = false,
): ScriptedBrain => {
  const recordedAsks: { input: string; isConstrained: boolean }[] = [];
  let index = 0;
  const brain: Brain = {
    ask: (input, options): Promise<Result<string, AiFailure>> => {
      const isConstrained = options?.schema !== undefined;
      if (isConstrained && refusesSchema)
        return Promise.resolve({
          ok: false,
          error: { kind: "unsupported-config", languages: [] },
        });
      recordedAsks.push({ input, isConstrained });
      const reply = replies[index] ?? makeAnswerReply("ran out of script");
      index += 1;
      return Promise.resolve({ ok: true, value: reply });
    },
    record: () => [],
  };
  return { brain, recordedAsks };
};

const makeEchoTool = (name = "echo"): Tool => ({
  name,
  description: "Echoes its text back.",
  parameters: { type: "object" } as never,
  // `unknown` in, so the type is checked rather than stringified: `String()`
  // on an object gives "[object Object]" and would pass silently.
  execute: (args) =>
    `echoed: ${typeof args.text === "string" ? args.text : ""}`,
});

describe("the agent loop", () => {
  test("calls a tool, feeds the result back, then answers", async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("echo", { text: "hi" }),
      makeAnswerReply("all done"),
    ]);
    const events: AgentEvent[] = [];
    const runResult = await runAgent(
      {
        brain,
        tools: [makeEchoTool()],
        onEvent: (event) => events.push(event),
      },
      "do it",
    );

    expect(runResult.ok).toBe(true);
    if (!runResult.ok) return;
    expect(runResult.value.text).toBe("all done");
    expect(runResult.value.steps).toBe(2);
    expect(runResult.value.constrained).toBe(true);
    // The tool's output is what the model saw on the second turn.
    expect(recordedAsks[1]?.input).toContain("Result of echo:");
    expect(recordedAsks[1]?.input).toContain("echoed: hi");
    expect(events.map((event) => event.kind)).toEqual([
      "step",
      "thought",
      "tool",
      "result",
      "step",
      "thought",
      "answer",
    ]);
  });

  test("a tool that throws comes back as text the model can act on", async () => {
    const angryTool: Tool = {
      name: "angry",
      description: "Always fails.",
      parameters: { type: "object" } as never,
      execute: () => {
        throw new Error("the disk is on fire");
      },
    };
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("angry"),
      makeAnswerReply("noted"),
    ]);
    const runResult = await runAgent({ brain, tools: [angryTool] }, "try it");

    expect(runResult.ok).toBe(true);
    // Handed back, not thrown: the run continues and the model is told why.
    expect(recordedAsks[1]?.input).toContain("ERROR: the disk is on fire");
  });

  test("an unknown tool is named back, with the ones that exist", async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("nosuch"),
      makeAnswerReply("fine"),
    ]);
    await runAgent({ brain, tools: [makeEchoTool()] }, "try it");
    expect(recordedAsks[1]?.input).toContain(
      'there is no tool called "nosuch"',
    );
    expect(recordedAsks[1]?.input).toContain("Tools: echo");
  });

  test("a tool that needs approval is denied when nobody can approve", async () => {
    const guardedTool: Tool = {
      ...makeEchoTool("guarded"),
      needsApproval: true,
    };
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("guarded"),
      makeAnswerReply("understood"),
    ]);
    const runResult = await runAgent({ brain, tools: [guardedTool] }, "try it");

    expect(runResult.ok).toBe(true);
    // No approver means no approval, rather than a default yes.
    expect(recordedAsks[1]?.input).toContain("DENIED");
  });

  test("an approver can let it through, and sees the arguments", async () => {
    const guardedTool: Tool = {
      ...makeEchoTool("guarded"),
      needsApproval: true,
    };
    const seenApprovals: Record<string, unknown>[] = [];
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("guarded", { text: "ok" }),
      makeAnswerReply("done"),
    ]);
    await runAgent(
      {
        brain,
        tools: [guardedTool],
        approve: (tool, args) => {
          seenApprovals.push({ name: tool.name, ...args });
          return true;
        },
      },
      "try it",
    );
    expect(seenApprovals).toEqual([{ name: "guarded", text: "ok" }]);
    expect(recordedAsks[1]?.input).toContain("echoed: ok");
  });

  test("a long result is cut before it goes back", async () => {
    const longTool: Tool = {
      ...makeEchoTool("long"),
      execute: () => "x".repeat(500),
    };
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("long"),
      makeAnswerReply("done"),
    ]);
    await runAgent({ brain, tools: [longTool], maxResultChars: 100 }, "try it");
    const fedInput = recordedAsks[1]?.input ?? "";
    expect(fedInput).toContain("cut by the harness");
    expect(fedInput.length).toBeLessThan(400);
  });

  test("the same call with the same answer is named as a repeat, not fed back again", async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("echo", { text: "x" }),
      makeToolCallReply("echo", { text: "x" }),
      makeAnswerReply("fine"),
    ]);
    const events: AgentEvent[] = [];
    await runAgent(
      {
        brain,
        tools: [makeEchoTool()],
        onEvent: (event) => events.push(event),
      },
      "loop a bit",
    );

    const statuses = events
      .filter((event) => event.kind === "result")
      .map((event) => event.status);
    expect(statuses).toEqual(["ok", "repeat"]);
    // The second time it is told so, instead of being handed the same text.
    expect(recordedAsks[2]?.input).toContain("Calling it again will not help");
  });

  test("a tool that answers differently each time is not a repeat", async () => {
    let calls = 0;
    const clockTool: Tool = {
      ...makeEchoTool("clock"),
      execute: () => `tick ${(calls += 1)}`,
    };
    const { brain } = makeScriptedBrain([
      makeToolCallReply("clock"),
      makeToolCallReply("clock"),
      makeAnswerReply("fine"),
    ]);
    const events: AgentEvent[] = [];
    await runAgent(
      { brain, tools: [clockTool], onEvent: (event) => events.push(event) },
      "read the clock twice",
    );

    const statuses = events
      .filter((event) => event.kind === "result")
      .map((event) => event.status);
    // Same arguments, different answers: legitimate, and left alone.
    expect(statuses).toEqual(["ok", "ok"]);
  });

  test("a third identical call stops the run, with the call named", async () => {
    const { brain } = makeScriptedBrain([
      makeToolCallReply("echo"),
      makeToolCallReply("echo"),
      makeToolCallReply("echo"),
      makeAnswerReply("never reached"),
    ]);
    const runResult = await runAgent(
      { brain, tools: [makeEchoTool()], maxSteps: 20 },
      "loop",
    );
    expect(runResult.ok).toBe(false);
    if (runResult.ok || runResult.error.kind !== "failed") return;
    // Early and specific, rather than burning every step to say the same thing.
    expect(runResult.error.detail).toContain("echo(");
    expect(runResult.error.detail).toContain("3 times");
  });

  test("a repeated failure is a loop too, which is the one that happens", async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      makeToolCallReply("nosuch"),
      makeToolCallReply("nosuch"),
      makeAnswerReply("fine"),
    ]);
    const events: AgentEvent[] = [];
    await runAgent(
      {
        brain,
        tools: [makeEchoTool()],
        onEvent: (event) => events.push(event),
      },
      "ask for a ghost",
    );

    const statuses = events
      .filter((event) => event.kind === "result")
      .map((event) => event.status);
    expect(statuses).toEqual(["error", "repeat"]);
    expect(recordedAsks[2]?.input).toContain("Calling it again will not help");
  });

  test("running out of steps is a failure with a reason, not a hang", async () => {
    // Different arguments each time, so this is slow progress rather than the
    // stuck detector above; the bound is what ends it.
    const { brain } = makeScriptedBrain([
      makeToolCallReply("echo", { text: "a" }),
      makeToolCallReply("echo", { text: "b" }),
      makeToolCallReply("echo", { text: "c" }),
    ]);
    const runResult = await runAgent(
      { brain, tools: [makeEchoTool()], maxSteps: 3 },
      "work slowly",
    );
    expect(runResult.ok).toBe(false);
    if (runResult.ok) return;
    expect(runResult.error.kind).toBe("failed");
    if (runResult.error.kind === "failed")
      expect(runResult.error.detail).toContain("within 3 steps");
  });

  test("output outside the shape costs a step and is asked for again", async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      "I think I will just chat instead.",
      makeAnswerReply("sorry, done"),
    ]);
    const runResult = await runAgent(
      { brain, tools: [makeEchoTool()] },
      "do it",
    );
    expect(runResult.ok).toBe(true);
    if (runResult.ok) expect(runResult.value.steps).toBe(2);
    expect(recordedAsks[1]?.input).toContain("not the shape asked for");
  });

  test('"call a tool" with no name is a shape error, not a missing tool', async () => {
    const { brain, recordedAsks } = makeScriptedBrain([
      makeStepReply({ action: "tool", tool: "" }),
      makeAnswerReply("recovered"),
    ]);
    const runResult = await runAgent(
      { brain, tools: [makeEchoTool()] },
      "do it",
    );
    expect(runResult.ok).toBe(true);
    // `granite4:350m` emitted exactly this, and being handed a list of tool
    // names told it nothing it could use.
    expect(recordedAsks[1]?.input).toContain("not the shape asked for");
    expect(recordedAsks[1]?.input).not.toContain("there is no tool called");
  });

  test("a refused schema drops the loop into prose, and it still finishes", async () => {
    // `claude -p` refuses schemas, measured. The refusal is the signal, and the
    // object is read out of the text instead.
    const { brain, recordedAsks } = makeScriptedBrain(
      [
        `Sure! Here you go:\n\`\`\`json\n${makeAnswerReply("read from prose")}\n\`\`\``,
      ],
      true,
    );
    const runResult = await runAgent(
      { brain, tools: [makeEchoTool()] },
      "do it",
    );

    expect(runResult.ok).toBe(true);
    if (!runResult.ok) return;
    expect(runResult.value.text).toBe("read from prose");
    expect(runResult.value.constrained).toBe(false);
    // Asked once, unconstrained: the refusal did not cost a step.
    expect(recordedAsks).toHaveLength(1);
    expect(recordedAsks[0]?.isConstrained).toBe(false);
  });

  test("a refusal that is not about the schema ends the run", async () => {
    const brain: Brain = {
      ask: () =>
        Promise.resolve({
          ok: false,
          error: { kind: "busy", detail: "another turn" },
        }),
      record: () => [],
    };
    const runResult = await runAgent({ brain, tools: [] }, "do it");
    expect(runResult.ok).toBe(false);
    if (!runResult.ok) expect(runResult.error.kind).toBe("busy");
  });

  test("an empty answer is asked for again rather than returned", async () => {
    const { brain } = makeScriptedBrain([
      makeAnswerReply("   "),
      makeAnswerReply("really done"),
    ]);
    const runResult = await runAgent({ brain, tools: [] }, "do it");
    expect(runResult.ok && runResult.value.text).toBe("really done");
  });

  test("on a real session, through the adapter", async () => {
    // The mock honours a schema when it is given a reply for one, so this is
    // the loop over the library's own session rather than over a stub brain.
    const access = await makeMockProvider({
      delayMs: 1,
      schemaReply: makeAnswerReply("from the mock"),
    }).access();
    if (access.kind !== "ready") throw new Error("expected ready");
    const sessionResult = await access.open();
    if (!sessionResult.ok) throw new Error("expected a session");

    const runResult = await runAgent(
      { brain: makeSessionBrain(sessionResult.value), tools: [] },
      "say something",
    );
    expect(runResult.ok && runResult.value.text).toBe("from the mock");
    // The turn went through the session, so it is in the session's record.
    expect(sessionResult.value.history()).toHaveLength(2);
    sessionResult.value.close();
  });
});
