# modelpact-agent

[![npm](https://img.shields.io/npm/v/modelpact-agent)](https://www.npmjs.com/package/modelpact-agent)
[![ci](https://github.com/AvdienkoSergey/modelpact-agent/actions/workflows/ci.yml/badge.svg?event=pull_request)](https://github.com/AvdienkoSergey/modelpact-agent/actions/workflows/ci.yml)
![node: ≥22](https://img.shields.io/badge/node-%E2%89%A522-339933)
[![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**A tool-calling loop over anything that answers a turn and keeps a record.
Ask, call a tool, feed the result back, ask again — bounded, gated, and
readable while it runs.**

## Install

```sh
npm install modelpact-agent modelpact
```

```ts
import { runAgent, makeSessionBrain, readFileTool } from "modelpact-agent";
import { makeOllamaProvider } from "modelpact-providers";

const access = await makeOllamaProvider({ model: "qwen3:14b" }).access();
if (access.kind !== "ready") return;
const opened = await access.open();
if (!opened.ok) return;

const run = await runAgent(
  { brain: makeSessionBrain(opened.value), tools: [readFileTool(workspace)] },
  "Read colour.txt and tell me the colour it names.",
);
```

## Why this is a storey and not a bigger session

The sibling package,
[modelpact-orchestrator](https://github.com/AvdienkoSergey/modelpact-orchestrator),
learned this the expensive way: a policy that picks between two models was
first built as a `ModelBackend`, and every guarantee a session makes leaked
through it. The rule that came out of it holds here too. A transport reaches
one model. A session holds one conversation with it. **A loop over turns is
neither**, and it goes above both, as an ordinary consumer.

So `runAgent` opens nothing, closes nothing, and owns no conversation. It is
handed a `Brain` — two methods, `ask` and `record` — and loops.

| Brain           | Is                                | Adapter                     |
| --------------- | --------------------------------- | --------------------------- |
| one model       | `AiSession`                       | `makeSessionBrain(session)` |
| several, routed | anything shaped like `AskingChat` | `makeChatBrain(chat)`       |

`AskingChat` is declared here rather than imported, which is why this package
depends on no router at all: an orchestrator satisfies it without being told
this type exists, and so does a product's own.

## What came from the reference agent

[HowProgrammingWorks/Agent](https://github.com/HowProgrammingWorks/Agent) is a
coding agent in Node: one OpenAI-compatible endpoint, a directory of tools, and
a loop. Its **shape** is the right shape, and it is the shape here:

- a step-limited loop, `for step = 1..maxSteps`, that fails with a reason
  rather than running forever;
- tool errors returned **to the model** as `ERROR: …` instead of thrown, so a
  model that gets its arguments wrong can fix them;
- a denial that reads the same way, `DENIED: …`, so refusing is a fact the
  model can act on rather than a hole;
- results truncated before they go back;
- an approval gate in front of anything that acts;
- events emitted per step, so a terminal can show the work.

## What could not come, and what replaced it

**Native tool calls.** That agent reads `message.tool_calls` off the response,
because its endpoint speaks the OpenAI protocol. When this loop was written the
contract had no tool protocol at all, so a call became a shape instead:

```json
{
  "reason": "...",
  "action": "tool",
  "tool": "readFile",
  "args": { "path": "x" },
  "answer": ""
}
```

Where the schema is honoured the model can only answer in it. Where it is
refused — `claude -p` refuses, measured — the same question is asked again in
prose and the object is read out of the text. The refusal is not an obstacle
here; it is the signal that picks the mode, and `AgentRun.constrained` reports
which one ran.

`modelpact` has since grown `ModelRequest.tools`, and a session opened with
them executes a call inside the turn — one model call per tool instead of two,
and a real `tool` role in the record. Teaching this loop a third mode that
opens such a session and falls back to the two above is the next thing it
wants, and the fallback is already written.

**A tool result is a user turn.** `AiRole` is `user` or `assistant`, so there
is nowhere for a third role. The result goes back as the next user message, and
that is the whole of the mapping.

**A tool that only pretends.** The first version of `writeNote` declared that
it needed approval and then wrote nothing, returning `would append …`. The gate
exists so that a tool which really acts can be shipped safely, and one that
asks permission and then does nothing demonstrates the opposite. It also lied
twice over: the model was told the line was appended, and could report as much
back. It appends now, contained by the same check the readers use.

**The path-token parser.** That agent resolves every path out of a shell
command and refuses what leaves the trust root, which takes a hundred lines
because its `bash` tool can reach anywhere. These tools take a path and nothing
else, so containment is a `resolve` and a prefix test. Shipping a shell tool
would mean shipping the hundred lines too, and this package does not.

## Measured, not assumed

**Every field of the step schema is required, and that is not tidiness.** Under
constrained decoding a model stops at what the schema demands. With `tool` and
`args` optional, `granite4:350m` answered `{"reason":"…","action":"tool"}` and
nothing else — an action with no verb. With all five required it answered
`{"reason":"…","action":"tool","tool":"listFiles","args":{},"answer":""}`.
`qwen3:14b` was the control and got it right either way.

**A small model loops, and the harness has to notice.** `granite4:350m` fills
the schema correctly and picks a real tool, and then cannot carry the task
across turns: handed a directory listing it asked for the same listing again,
and told that repeating would not help it repeated once more. A step limit
alone turned that into eight wasted steps and the word `failed`. Now an
identical call with an identical answer — of any status, because the repeat
that actually happened was a repeated error — is named back as a repeat, and a
third one stops the run and says which call it was. Six steps and a diagnosis
instead of eight and a shrug. It does not make the model capable; it makes the
failure legible.

**The answer is the last object in the reply, not the widest braces in it.**
Unconstrained — which is what an orchestrator's local side was until
`Orchestrator.ask` learned to pass a schema through — `granite4.2:3b-q8_0`
answered a write step with a correct object, then 30 000 characters of second thoughts about it,
then a corrected object. Reading from the first `{` to the last `}` spans all
three and parses as none, so a reply carrying two usable answers was thrown
away as prose and the run ended having written nothing. The scan is now
string-aware, because the argument of a write tool is a whole file and a brace
counter that does not skip quoted text closes the object on the first `}` in
the code it carries; the last object that is a step wins, since a draft before
it is not the decision. Measured 2026-09-12.

**"Done" is a report, and only the caller can check it.** The same run then
answered `Please provide the requested shape as a JSON object` — a non-empty
string, so the loop returned it as a success. The loop cannot tell a finished
task from a model that gave up; what the task was for lives with whoever called
`runAgent`. `acceptAnswer` is that vote: `true` accepts, and a string is the
objection put to the model, which costs a step and lets it carry on with the
steps it has left. The caller that translates OCaml uses it to run the type
checker before it believes anything.

**A repeat is counted per call, and the stop says a number that is true.**
One counter for the whole run called it stuck when two _different_ calls each
repeated once, which is a model going round a wider circle rather than one
jammed on a tool — and the message then named the last call with the counter's
number rather than that call's: `readFile(lib.ml)` reported as called three
times when it had been called twice. Worse, the stop landed on exactly the step
a router would have escalated on, so the cloud side never got the turn that
would have finished the work. Hence `stuckLimit`, the caller's to raise above
its router's own threshold. Measured 2026-09-12.

**A barren reply repeats too, and only tool calls were watched.** After the
work was done and the type checker green, `granite4.2:3b-q8_0` answered
`{"reason":"…","action":"answer","answer":""}` seven times word for word, each
time told the answer was empty, until the step limit ended the run as a
failure. The same prompt gets the same reply out of a model at rest, so a reply
identical to the last one that ran nothing and finished nothing is a loop by
the same rule as an identical call, and it is counted by the same limit.
Measured 2026-09-12.

**The truncation default is 4 000 characters, not 60 000.** That agent talks to
a 200k-token cloud model. A local 4096-token window is filled by one file read
at 60k, so the small default is the one that keeps a local model working.

## Tests

[`src/agent.test.ts`](src/agent.test.ts) runs the loop on a brain made of
strings, because what is under test is control flow: what reaches the model
after a tool, what a denial and a throw look like from inside, where the loop
stops. [`src/tools.test.ts`](src/tools.test.ts) covers the one part of a tool
that has to be right, which is not leaving the workspace.
[`src/live.test.ts`](src/live.test.ts) runs the real loop on a real model with
a real tool, and skips loudly without a daemon.

```sh
npm test
npm run chat    # a terminal agent; AGENT_MODEL, ROOT, APPROVE
```

The prose branch — a brain that refuses schemas — is reached by a stub here.
No transport in the family refuses any more: `claude -p` takes `--json-schema`
from 2.1.236 on, so the branch is kept for a backend that cannot constrain
decoding, and for an older CLI, rather than exercised live.

## Scripts

| Script                  | What it does                                              |
| ----------------------- | --------------------------------------------------------- |
| `npm run typecheck`     | `tsc --noEmit` over `src`                                 |
| `npm run lint`          | ESLint, type-aware                                        |
| `npm run format:check`  | Prettier, check only                                      |
| `npm test`              | Vitest; the live suite skips without a daemon             |
| `npm run check:surface` | builds, then reads the declarations without `@types/node` |
| `npm run build`         | `dist/` — JS, declarations, maps                          |

## Releases

Versions come from [conventional commits](https://www.conventionalcommits.org)
by way of release-please, and are published to npm from CI by trusted
publishing. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
