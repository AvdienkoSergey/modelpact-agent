/**
 * What the agent thinks with: a turn, and the record that turn joined.
 *
 * Two lines each, because the library already does the work. A session is one
 * model's; a router is several behind a policy. The loop above cares about
 * neither, only that asking a question appends a turn somewhere and the record
 * comes back — which is why the second adapter takes a shape rather than a
 * package.
 */

import type {
  AiFailure,
  AiMessage,
  AiSession,
  JsonSchema,
  Result,
} from "modelpact";

export interface AskOptions {
  /** Constrained decoding, honoured or refused; the agent reads the refusal and adapts. */
  readonly schema?: JsonSchema;
}

/**
 * Anything on the storey above a session that answers a turn and keeps the
 * record: `modelpact-orchestrator` is one, a product's own router is another.
 *
 * Declared here rather than imported, so this package depends on no particular
 * one. The loop never needed more than these two methods, and an orchestrator
 * satisfies them without being told about this type.
 */
export interface AskingChat {
  readonly ask: (
    input: string,
    options?: AskOptions,
  ) => Promise<Result<{ readonly text: string }, AiFailure>>;
  readonly record: () => readonly AiMessage[];
}

export interface Brain {
  readonly ask: (
    input: string,
    options?: AskOptions,
  ) => Promise<Result<string, AiFailure>>;
  readonly record: () => readonly AiMessage[];
}

export const makeSessionBrain = (session: AiSession): Brain => ({
  ask: (input, options) =>
    session.prompt(
      input,
      options?.schema === undefined ? {} : { schema: options.schema },
    ),
  record: () => session.history(),
});

export const makeChatBrain = (chat: AskingChat): Brain => ({
  ask: async (input, options) => {
    const answerResult = await chat.ask(
      input,
      options?.schema === undefined ? {} : { schema: options.schema },
    );
    return answerResult.ok
      ? { ok: true, value: answerResult.value.text }
      : answerResult;
  },
  record: () => chat.record(),
});
