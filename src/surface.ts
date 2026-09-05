/**
 * Every exported name, read from the *emitted* declarations the way a consumer
 * without `@types/node` reads them: `types: []`, `skipLibCheck` off. A node
 * type in a public signature fails here; a node import in a function body does
 * not, because it never reaches the `.d.ts` — which is what lets the tools
 * touch the file system and still ship a surface a browser build can compile.
 */
import {
  listFilesTool,
  makeChatBrain,
  makeSessionBrain,
  readFileTool,
  runAgent,
  writeNoteTool,
  type AgentEvent,
  type AgentParts,
  type AgentRun,
  type AskingChat,
  type AskOptions,
  type Brain,
  type ResultStatus,
  type Tool,
} from "../dist/index.js";

export const values = {
  listFilesTool,
  makeChatBrain,
  makeSessionBrain,
  readFileTool,
  runAgent,
  writeNoteTool,
};
export type Types = [
  AgentEvent,
  AgentParts,
  AgentRun,
  AskingChat,
  AskOptions,
  Brain,
  ResultStatus,
  Tool,
];
