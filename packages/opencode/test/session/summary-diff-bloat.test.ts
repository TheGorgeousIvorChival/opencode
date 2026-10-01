/**
 * Regression test: `summarize()` must not append a duplicate copy of a turn's
 * (cumulative, ever-growing) diff to the append-only event log on every step.
 *
 * `summarize` runs after each step and recomputes the turn diff from the same
 * step-start/step-finish snapshots. Because the diff is recomputed rather than
 * mutated, most calls produce a summary identical to the one already stored.
 * Publishing it anyway stored a full copy of the diff per step in the `event`
 * table, which grows quadratically with turn length -- a single long turn was
 * observed to accumulate ~710MB across 279 `message.updated` events.
 *
 * The fix skips the publish when the recomputed diff deep-equals the stored one.
 * The projector upserts `message` rows, so an unchanged summary is a no-op.
 */
import { expect } from "bun:test"
import { Effect, Layer } from "effect"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import fs from "fs/promises"
import path from "path"
import { Database } from "@opencode-ai/core/database/database"
import { Session } from "@/session/session"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionSummary } from "../../src/session/summary"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { EventV2 } from "@opencode-ai/core/event"
import { EventTable } from "@opencode-ai/core/event/sql"
import { like } from "drizzle-orm"
import { provideTmpdirServer } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestLLMServer } from "../lib/llm-server"

import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { RuntimeFlags } from "@/effect/runtime-flags"

const mcp = Layer.succeed(
  MCP.Service,
  MCP.Service.of({
    status: () => Effect.succeed({}),
    clients: () => Effect.succeed([]),
    instructions: () => Effect.succeed([]),
    tools: () => Effect.succeed([]),
    prompts: () => Effect.succeed([]),
    resources: () => Effect.succeed([]),
    resourceTemplates: () => Effect.succeed([]),
    add: () => Effect.succeed({ status: "disabled" as const }),
    connect: () => Effect.void,
    disconnect: () => Effect.void,
    getPrompt: () => Effect.succeed(undefined),
    readResource: () => Effect.succeed(undefined),
    startAuth: () => Effect.die("unexpected MCP auth"),
    authenticate: () => Effect.die("unexpected MCP auth"),
    finishAuth: () => Effect.die("unexpected MCP auth"),
    removeAuth: () => Effect.void,
    supportsOAuth: () => Effect.succeed(false),
    hasStoredTokens: () => Effect.succeed(false),
    getAuthStatus: () => Effect.succeed("not_authenticated" as const),
  }),
)

const lsp = Layer.succeed(
  LSP.Service,
  LSP.Service.of({
    init: () => Effect.void,
    status: () => Effect.succeed([]),
    hasClients: () => Effect.succeed(false),
    touchFile: () => Effect.void,
    diagnostics: () => Effect.succeed([]),
    hover: () => Effect.succeed(undefined),
    definition: () => Effect.succeed(undefined),
    references: () => Effect.succeed([]),
    implementation: () => Effect.succeed(undefined),
    documentSymbol: () => Effect.succeed([]),
    workspaceSymbol: () => Effect.succeed([]),
    prepareCallHierarchy: () => Effect.succeed([]),
    incomingCalls: () => Effect.succeed([]),
    outgoingCalls: () => Effect.succeed([]),
  }),
)

const root = LayerNode.group([
  SessionPrompt.node,
  Session.node,
  SessionProjector.node,
  SessionSummary.node,
  Database.node,
  EventV2.node,
  CrossSpawnSpawner.node,
  LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] }),
])

const it = testEffect(
  LayerNode.compile(root, [
    [MCP.node, mcp],
    [LSP.node, lsp],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
  ]),
)

const providerCfg = (url: string) => ({
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: url,
      },
    },
  },
})

it.live("repeated summarize() calls with an unchanged diff do not append duplicate events", () =>
  provideTmpdirServer(
    Effect.fnUntraced(function* ({ dir, llm }) {
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const summary = yield* SessionSummary.Service
      const database = yield* Database.Service
      const { db } = database

      const session = yield* sessions.create({
        title: "summary dedup test",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })

      const command = `echo 'summary dedup test content' > ${path.join(dir, "dedup-test.txt")}`
      yield* llm.toolMatch((hit) => JSON.stringify(hit.body).includes("create the file"), "bash", { command })
      yield* llm.textMatch((hit) => JSON.stringify(hit.body).includes("bash"), "done")

      yield* prompt.prompt({
        sessionID: session.id,
        agent: "build",
        noReply: true,
        parts: [{ type: "text", text: "create the file" }],
      })

      yield* prompt.loop({ sessionID: session.id })

      const allMsgs = yield* MessageV2.filterCompactedEffect(session.id)
      const user = allMsgs.find(
        (msg): msg is SessionV1.WithParts & { info: SessionV1.User } => msg.info.role === "user",
      )
      if (!user) throw new Error("Expected user message")

      // Wait for the first summarize() to land a non-empty diff.
      let diff: Array<{ file?: string }> = []
      for (let i = 0; i < 50; i++) {
        diff = yield* summary.diff({ sessionID: session.id, messageID: user.info.id })
        if (diff.length > 0) break
        yield* Effect.sleep("100 millis")
      }
      expect(diff.length).toBeGreaterThan(0)

      const countSummaryEvents = Effect.fnUntraced(function* () {
        const rows = yield* db
          .select({ id: EventTable.id })
          .from(EventTable)
          .where(like(EventTable.type, "message.updated%"))
          .all()
          .pipe(Effect.orDie)
        return rows.length
      })

      const before = yield* countSummaryEvents()

      // Re-run summarize several times. The snapshots have not changed, so the
      // recomputed diff is identical to the stored one every time and none of
      // these calls should append a new event.
      for (let i = 0; i < 5; i++) {
        yield* summary.summarize({ sessionID: session.id, messageID: user.info.id })
      }

      const after = yield* countSummaryEvents()
      expect(after).toBe(before)

      // The diff must still be readable -- dedup must not drop the summary.
      const stillThere = yield* summary.diff({ sessionID: session.id, messageID: user.info.id })
      expect(stillThere.length).toBe(diff.length)

      // And the file the turn produced still exists, so we did not skip real work.
      const exists = yield* Effect.promise(() =>
        fs
          .access(path.join(dir, "dedup-test.txt"))
          .then(() => true)
          .catch(() => false),
      )
      expect(exists).toBe(true)
    }),
    { git: true, config: providerCfg },
  ),
)
