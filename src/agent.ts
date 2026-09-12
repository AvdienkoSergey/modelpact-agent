/**
 * The loop: ask, call a tool, feed the result back, ask again.
 *
 * Storey three, and thicker than the router that lived here before. The shape
 * is the one from HowProgrammingWorks/Agent — a step-limited loop, tool errors
 * returned to the model rather than thrown, results truncated before they are
 * fed back, an approval gate in front of anything that acts. What is different
 * is everything below the loop: that agent talks to one OpenAI-compatible
 * endpoint and reads native `tool_calls`; this one talks to whatever a `Brain`
 * is, and this contract has no tool protocol at all.
 *
 * So a call is a constrained answer instead. The contract does have schemas,
 * honoured or refused and never ignored, and a tool call is a shape. Where the
 * schema is honoured the model can only answer in it; where it is refused —
 * `claude -p` refuses, measured — the same question is asked in prose and the
 * object is read out of the text. The refusal is not an obstacle here, it is
 * the signal that picks the mode.
 */

import {
  jsonSchema,
  type AiFailure,
  type JsonSchema,
  type Result,
} from "modelpact";

import type { Brain } from "./brain.js";

export interface Tool {
  readonly name: string;
  readonly description: string;
  /** The argument shape, shown to the model in the prompt. */
  readonly parameters: JsonSchema;
  /** Denied unless `approve` says otherwise; the default answer is no. */
  readonly needsApproval?: boolean;
  readonly execute: (args: Record<string, unknown>) => Promise<string> | string;
}

/** `repeat` is `ok` that the model has already been given once, word for word. */
export type ResultStatus = "ok" | "error" | "denied" | "repeat";

export type AgentEvent =
  | { readonly kind: "step"; readonly step: number; readonly maxSteps: number }
  | { readonly kind: "thought"; readonly step: number; readonly reason: string }
  | {
      readonly kind: "tool";
      readonly step: number;
      readonly name: string;
      readonly args: Record<string, unknown>;
    }
  | {
      readonly kind: "result";
      readonly step: number;
      readonly name: string;
      readonly status: ResultStatus;
      readonly preview: string;
    }
  | { readonly kind: "answer"; readonly step: number; readonly text: string };

export interface AgentParts {
  readonly brain: Brain;
  readonly tools: readonly Tool[];
  /** The loop is bounded, and running out is a failure with a reason, not a hang. */
  readonly maxSteps?: number;
  readonly instructions?: string;
  /**
   * A tool result longer than this is cut before it goes back to the model.
   * The default is small on purpose: the reference agent allows 60k characters
   * because it talks to a 200k-token cloud model, and one file read at that
   * size fills a 4096-token local window on its own.
   */
  readonly maxResultChars?: number;
  readonly approve?: (
    tool: Tool,
    args: Record<string, unknown>,
  ) => Promise<boolean> | boolean;
  /**
   * Whether an answer ends the run. `true` accepts it; a string is the
   * objection put to the model, which costs a step and the loop goes on.
   *
   * A model that says it is done is reporting, not proving, and the loop
   * cannot tell the difference on its own — only the caller knows what the
   * task was for. `granite4.2:3b-q8_0` finished a translation run with
   * "Please provide the requested shape as a JSON object" and no file written,
   * and the run was a success because the string was not empty. Measured
   * 2026-09-12.
   */
  readonly acceptAnswer?: (
    text: string,
  ) => Promise<true | string> | true | string;
  /**
   * How many times one call may give the same answer, or one reply may repeat
   * itself without moving, before the run is stopped. Two by default: one is a
   * slip and gets a nudge, and every step past the second spends tokens to be
   * told the same thing.
   *
   * Raise it where something under the loop can rescue a stuck turn. A router
   * escalates on the ask *after* the stuck one, so a limit that stops at the
   * same moment denies the rescue: `granite4.2:3b-q8_0` re-read two files, the
   * loop stopped at the second repeat, and the cloud side never got the turn
   * that would have finished the work. A limit above the router's own
   * threshold is what leaves room for it.
   */
  readonly stuckLimit?: number;
  readonly onEvent?: (event: AgentEvent) => void;
}

export interface AgentRun {
  readonly text: string;
  readonly steps: number;
  /** False when the brain refused the schema and the loop read prose instead. */
  readonly constrained: boolean;
}

const DEFAULTS = { maxSteps: 12, maxResultChars: 4_000, stuckLimit: 2 };

const INSTRUCTIONS =
  "You are a careful assistant with tools. Work in small steps: call one tool, read its result, then decide. Answer only when the task is done or cannot be done.";

/**
 * Every field is required, which is not tidiness.
 *
 * Under constrained decoding a model stops at what the schema demands, so an
 * optional `tool` is simply never emitted: `granite4:350m` answered
 * `{"action":"tool"}` with no tool name until all five were required, and then
 * answered `{"action":"tool","tool":"listFiles","args":{},"answer":""}`.
 * Measured on 2026-09-03, against `qwen3:14b` as the control.
 */
const STEP_SHAPE = {
  type: "object",
  properties: {
    reason: { type: "string" },
    action: { type: "string", enum: ["tool", "answer"] },
    tool: { type: "string" },
    args: { type: "object" },
    answer: { type: "string" },
  },
  required: ["reason", "action", "tool", "args", "answer"],
  additionalProperties: false,
};

const STEP_SCHEMA: JsonSchema = (() => {
  const builtSchema = jsonSchema(STEP_SHAPE);
  if (builtSchema === null) throw new Error("the step shape is not a schema");
  return builtSchema;
})();

interface Step {
  readonly reason: string;
  readonly action: "tool" | "answer";
  readonly tool: string;
  readonly args: Record<string, unknown>;
  readonly answer: string;
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

const asText = (value: unknown): string =>
  typeof value === "string" ? value : "";

const parseObject = (text: string): Record<string, unknown> | null => {
  try {
    return asRecord(JSON.parse(text));
  } catch {
    return null;
  }
};

/**
 * Every `{…}` in the text that balances, in the order they open.
 *
 * String-aware, and it has to be: the argument of a write tool is a whole
 * file, and a brace counter that does not skip quoted text closes the object
 * on the first `}` in the code it carries.
 */
const findObjectSpans = (text: string): string[] => {
  const spans: string[] = [];
  let openedAt = -1;
  let depth = 0;
  let isInString = false;
  let isEscaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (isInString) {
      if (isEscaped) isEscaped = false;
      else if (character === "\\") isEscaped = true;
      else if (character === '"') isInString = false;
      continue;
    }
    if (character === '"') isInString = true;
    else if (character === "{") {
      if (depth === 0) openedAt = index;
      depth += 1;
    } else if (character === "}" && depth > 0) {
      depth -= 1;
      if (depth === 0) spans.push(text.slice(openedAt, index + 1));
    }
  }
  return spans;
};

/**
 * The objects in a reply, last first, because the last is the model's final
 * word on what to do.
 *
 * Constrained, the whole text is the object and the first line here is the
 * whole of it. Unconstrained it is not: `granite4.2:3b-q8_0`, asked in prose
 * because the schema does not reach the local side of an orchestrator,
 * answered with a correct object, then 30k characters of second thoughts about
 * it, then a corrected object. The widest braces span all three, so the reply
 * that carried two usable answers parsed as none. Measured 2026-09-12.
 */
const readObjects = (text: string): readonly Record<string, unknown>[] => {
  const wholeObject = parseObject(text.trim());
  if (wholeObject !== null) return [wholeObject];
  const balancedObjects = findObjectSpans(text)
    .map(parseObject)
    .filter((object): object is Record<string, unknown> => object !== null)
    .reverse();
  if (balancedObjects.length > 0) return balancedObjects;
  // Nothing balanced, which is what a cut-off reply leaves: the widest braces
  // are the last thing to try before calling it prose.
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace < 0 || lastBrace <= firstBrace) return [];
  const widestObject = parseObject(text.slice(firstBrace, lastBrace + 1));
  return widestObject === null ? [] : [widestObject];
};

const toStep = (fields: Record<string, unknown>): Step | null => {
  const action = fields.action;
  if (action !== "tool" && action !== "answer") return null;
  // "call a tool" with no name is malformed output, not a call to a tool that
  // does not exist; the repair prompt is a better answer than a list of names.
  if (action === "tool" && asText(fields.tool).trim() === "") return null;
  return {
    reason: asText(fields.reason),
    action,
    tool: asText(fields.tool),
    args: asRecord(fields.args) ?? {},
    answer: asText(fields.answer),
  };
};

/** The last object that is a step: an example or a draft before it is not the decision. */
const readStep = (text: string): Step | null => {
  for (const fields of readObjects(text)) {
    const step = toStep(fields);
    if (step !== null) return step;
  }
  return null;
};

const errorText = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

const truncate = (text: string, maxChars: number): string =>
  text.length <= maxChars
    ? text
    : `${text.slice(0, maxChars)}\n…[cut by the harness]`;

/** A refusal of the schema, as against a refusal of the question. */
const isSchemaRefusal = (failure: AiFailure): boolean =>
  failure.kind === "unsupported-config" || failure.kind === "unsupported";

const describeTools = (tools: readonly Tool[]): string => {
  if (tools.length === 0) return "There are no tools. Answer directly.";
  const lines = tools.map(
    (tool) =>
      `- ${tool.name}: ${tool.description}\n  args: ${JSON.stringify(tool.parameters)}`,
  );
  return `Tools:\n${lines.join("\n")}`;
};

const SHAPE_REMINDER =
  'Answer with one JSON object and nothing else: {"reason": why, "action": "tool" or "answer", "tool": the name or "", "args": an object, "answer": the final text or ""}.';

const renderOpeningPrompt = (
  task: string,
  tools: readonly Tool[],
  instructions: string,
): string =>
  `${instructions}\n\n${describeTools(tools)}\n\n${SHAPE_REMINDER}\n\nTask: ${task}`;

/**
 * What a tool said the last time it was called with these exact arguments.
 *
 * A model that calls the same tool with the same arguments and is handed the
 * same answer has nothing new to go on, and small ones will do it until the
 * step limit stops them — `granite4:350m` read one file seven times. The tool
 * still runs, because a clock or a fetch is allowed to answer differently; it
 * is only when the answer is identical too that the loop is real, and then the
 * model is told so instead of being handed the same text again.
 */
type SeenResults = Map<string, string>;

const toCallKey = (step: Step): string =>
  `${step.tool}(${JSON.stringify(step.args)})`;

const runTool = async (
  step: Step,
  parts: AgentParts,
): Promise<{ status: ResultStatus; rendered: string }> => {
  const tool = parts.tools.find((candidate) => candidate.name === step.tool);
  if (tool === undefined) {
    const names = parts.tools.map((candidate) => candidate.name).join(", ");
    return {
      status: "error",
      rendered: `ERROR: there is no tool called "${step.tool}". Tools: ${names}`,
    };
  }
  if (tool.needsApproval === true) {
    // No approver means no approval: a tool that says it needs one does not
    // get to act because nobody was there to say no.
    const approved = (await parts.approve?.(tool, step.args)) ?? false;
    if (!approved)
      return {
        status: "denied",
        rendered: "DENIED: the user did not approve this call.",
      };
  }
  try {
    const toolOutput = await tool.execute(step.args);
    return {
      status: "ok",
      rendered: truncate(
        toolOutput,
        parts.maxResultChars ?? DEFAULTS.maxResultChars,
      ),
    };
  } catch (error) {
    // Handed back rather than thrown: a model that is told what went wrong can
    // fix its arguments, and a thrown error ends a run that could have finished.
    return { status: "error", rendered: `ERROR: ${errorText(error)}` };
  }
};

export async function runAgent(
  parts: AgentParts,
  task: string,
): Promise<Result<AgentRun, AiFailure>> {
  const maxSteps = parts.maxSteps ?? DEFAULTS.maxSteps;
  const emit = (event: AgentEvent): void => parts.onEvent?.(event);
  let isConstrained = true;

  const think = async (input: string): Promise<Result<string, AiFailure>> => {
    if (isConstrained) {
      const answerResult = await parts.brain.ask(input, {
        schema: STEP_SCHEMA,
      });
      if (answerResult.ok) return answerResult;
      if (!isSchemaRefusal(answerResult.error)) return answerResult;
      // The contract's own answer, put to use: honoured or refused, never
      // ignored. Refused means ask again in prose and read the object out.
      isConstrained = false;
    }
    return parts.brain.ask(input);
  };

  let input = renderOpeningPrompt(
    task,
    parts.tools,
    parts.instructions ?? INSTRUCTIONS,
  );

  const stuckLimit = parts.stuckLimit ?? DEFAULTS.stuckLimit;
  const seenResults: SeenResults = new Map();
  /**
   * Repeats counted per call, not once for the whole run.
   *
   * One counter for every call said "stuck" when two *different* calls each
   * repeated once, which is a model going round a wider circle, not a model
   * jammed on one tool — and it then named the last call and a number that
   * belonged to the counter rather than to that call: `readFile(lib.ml)`
   * reported as called three times when it had been called twice. Measured
   * 2026-09-12.
   */
  const repeatsByCall = new Map<string, number>();
  /**
   * A turn that ran nothing and finished nothing: an unreadable reply, an
   * empty answer, an answer the caller sent back. The same prompt gets the
   * same reply out of a model at rest, so a reply identical to the last barren
   * one is a loop — `granite4.2:3b-q8_0` answered
   * `{"action":"answer","answer":""}` seven times word for word, after the
   * work was done and verified, and nothing noticed because only tool calls
   * were watched. Measured 2026-09-12.
   */
  let barrenReply: string | null = null;
  let barrenRepeats = 0;

  const isBarrenLoop = (reply: string): boolean => {
    barrenRepeats = reply === barrenReply ? barrenRepeats + 1 : 0;
    barrenReply = reply;
    return barrenRepeats >= stuckLimit;
  };

  const barrenFailure = (): Result<never, AiFailure> => ({
    ok: false,
    error: {
      kind: "failed",
      detail: `the agent gave the same reply ${barrenRepeats + 1} times without calling a tool or finishing`,
    },
  });

  for (let stepNumber = 1; stepNumber <= maxSteps; stepNumber += 1) {
    emit({ kind: "step", step: stepNumber, maxSteps });
    const thoughtResult = await think(input);
    if (!thoughtResult.ok) return thoughtResult;

    const decidedStep = readStep(thoughtResult.value);
    if (decidedStep === null) {
      if (isBarrenLoop(thoughtResult.value)) return barrenFailure();
      // A repair costs a step, which keeps the bound honest.
      input = `That was not the shape asked for. ${SHAPE_REMINDER}`;
      continue;
    }
    emit({ kind: "thought", step: stepNumber, reason: decidedStep.reason });

    if (decidedStep.action === "answer") {
      const text = decidedStep.answer.trim();
      if (text === "") {
        if (isBarrenLoop(thoughtResult.value)) return barrenFailure();
        input =
          'The answer was empty. Give the final text in "answer", or call a tool.';
        continue;
      }
      const verdict = (await parts.acceptAnswer?.(text)) ?? true;
      if (verdict !== true) {
        if (isBarrenLoop(thoughtResult.value)) return barrenFailure();
        input = `${verdict} ${SHAPE_REMINDER}`;
        continue;
      }
      emit({ kind: "answer", step: stepNumber, text });
      return {
        ok: true,
        value: { text, steps: stepNumber, constrained: isConstrained },
      };
    }

    emit({
      kind: "tool",
      step: stepNumber,
      name: decidedStep.tool,
      args: decidedStep.args,
    });
    const toolOutcome = await runTool(decidedStep, parts);

    const callKey = toCallKey(decidedStep);
    // Any status, not only `ok`: a refusal repeated word for word is a loop as
    // surely as an answer repeated word for word, and it is the one that
    // actually happened.
    const isRepeat = seenResults.get(callKey) === toolOutcome.rendered;
    seenResults.set(callKey, toolOutcome.rendered);
    const repeatsHere = isRepeat ? (repeatsByCall.get(callKey) ?? 0) + 1 : 0;
    repeatsByCall.set(callKey, repeatsHere);
    // Work happened, whatever else is true: the barren run is broken.
    barrenReply = null;
    barrenRepeats = 0;

    emit({
      kind: "result",
      step: stepNumber,
      name: decidedStep.tool,
      status: isRepeat ? "repeat" : toolOutcome.status,
      preview: toolOutcome.rendered.slice(0, 200),
    });
    // The contract carries user and assistant turns and nothing else, so a
    // tool result goes back as the next user message. That is the whole of the
    // mapping, and it is why no tool protocol was needed in the contract.
    if (repeatsHere >= stuckLimit) {
      return {
        ok: false,
        error: {
          kind: "failed",
          detail: `the agent called ${callKey} ${repeatsHere + 1} times for the same answer and did not move on`,
        },
      };
    }

    input = isRepeat
      ? `${callKey} was called before and has just given exactly the same answer:\n${toolOutcome.rendered}\n\nCalling it again will not help. Use that answer now, or call a different tool, or finish. ${SHAPE_REMINDER}`
      : `Result of ${decidedStep.tool}:\n${toolOutcome.rendered}\n\n${SHAPE_REMINDER}`;
  }

  return {
    ok: false,
    error: {
      kind: "failed",
      detail: `the agent did not finish within ${maxSteps} steps`,
    },
  };
}
