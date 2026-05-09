# Long-Term Memory Implementation Guide

This guide walks you through implementing **vector embedding-based long-term memory** in CodersGPT. After this, the AI will remember user preferences, identity, and goals across conversations.

---

## How It Works (Architecture)

```
USER MESSAGE
    |
    v
+----------- LangGraph State Graph -----------+
|                                              |
|  START                                       |
|    |                                         |
|    +---> callLlm (parallel)                  |
|    |       |                                 |
|    |       1. Fetch memories from store       |
|    |       2. Inject into system prompt       |
|    |       3. Call LLM with tools            |
|    |       |                                 |
|    |       v                                 |
|    |     shouldContinue?                     |
|    |       |--- tool calls ---> toolNode     |
|    |       |--- no tools -----> END          |
|    |                                         |
|    +---> memoryRememberNode (parallel)       |
|            |                                 |
|            1. Fetch existing memories        |
|            2. Extract new facts via LLM      |
|            3. Store new memories (is_new)    |
|            |                                 |
|            v                                 |
|           END                                |
+----------------------------------------------+
```

**Key concept**: `callLlm` and `memoryRememberNode` run **in parallel**. Memory extraction does NOT slow down the response.

---

## Prerequisites

Make sure you have:
- PostgreSQL running with `DATABASE_URL` configured in `.env`
- `OPENAI_API_KEY` set in `.env` (used for both LLM calls and embeddings)
- The existing LangGraph chat pipeline working (START -> callLlm -> tools/END)

---

## Step 1: Install the OpenAI Package

```bash
pnpm add openai
```

**Why?** We use the raw OpenAI SDK (not LangChain) for memory extraction because LangChain's `withStructuredOutput` leaks structured output tokens into the streamed response. The raw SDK avoids this.

---

## Step 2: Create the Vector Store (`lib/store.ts`)

This file sets up a **PostgresStore** with OpenAI embeddings for vector-based memory storage and retrieval.

Create `lib/store.ts`:

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";

const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

const store = PostgresStore.fromConnString(process.env.DATABASE_URL!, {
  index: {
    dims: 1536,
    embed: embeddings,
  },
});

let setupDone = false;

export async function getStore() {
  if (!setupDone) {
    await store.setup();
    setupDone = true;
  }
  return store;
}
```

### What's happening here:
- **`OpenAIEmbeddings`** — Uses `text-embedding-3-small` model to convert text into 1536-dimensional vectors
- **`PostgresStore`** — LangGraph's built-in vector store that creates its own tables in your existing PostgreSQL database
- **`getStore()`** — Ensures `store.setup()` (which creates the DB tables) only runs once
- **Namespace pattern** — Memories are stored under `[userId, "memories"]`, isolating each user's data

---

## Step 3: Create Prompt Templates (`app/api/chat/prompts.ts`)

Two prompts are needed:
1. **System prompt** — The main prompt that includes user memories for personalization
2. **Memory extraction prompt** — Guides the LLM to extract facts from user messages

Create `app/api/chat/prompts.ts`:

```typescript
import { PromptTemplate } from "@langchain/core/prompts";

export const BASE_SYSTEM_PROMPT_TEMPLATE =
  PromptTemplate.fromTemplate(`You are CodersGPT — a powerful, full-scale AI conversational assistant designed for developers with memory capabilities.
If user-specific memory is available, use it to personalize
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored
assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..."
say "Since your project is built with TypeScript..."

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile.

## Response Philosophy
- **Be Direct**: Provide immediate, high-value answers. Avoid "As an AI..." or "I can help with that."
- **Brevity & Precision**: Keep responses concise and to the point. If a short answer suffices, do not write a long one.
- **Tone**: Professional, confident, and energetic. Use subtle technical emojis to make the chat feel alive.

## Formatting Standards
1. **Natural Markdown**: Use proper Markdown hierarchy (##, ###) ONLY when organizing complex, multi-part answers.
2. **Lists & Bullets**: Use bullet points or numbered lists for steps, features, or comparisons to ensure scannability.
3. **Code Blocks**: Always wrap code in fenced blocks with the correct language tag. Focus on production-ready, modern syntax.
4. **No Raw Text**: Always use bolding for key terms and maintain a clean, organized layout.

Always output structured, clean, and visually organized Markdown.

The user's memory (which may be empty) is provided as: {user_details_content}
`);

export const REMEMBER_MEMORY_PROMPT =
  PromptTemplate.fromTemplate(`You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, preferences, goals).
- Set is_new=true ONLY if it adds NEW info.
- Keep each memory as a short atomic sentence.`);
```

### What's happening here:
- **`PromptTemplate.fromTemplate()`** — LangChain's template system with `{placeholder}` syntax
- **`{user_details_content}`** — Gets replaced with the user's stored memories (or `"(empty)"` if none)
- **`REMEMBER_MEMORY_PROMPT`** — Shows existing memories to the LLM so it can avoid duplicates via `is_new`

---

## Step 4: Create the Structured Output Helper (`http/llm.ts`)

This file uses the **raw OpenAI SDK** to extract structured memory decisions from user messages.

Create the `http/` directory and `http/llm.ts`:

```typescript
import { z } from "zod";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { SystemMessage, BaseMessage } from "@langchain/core/messages";

const openai = new OpenAI();

const MemoryItemSchema = z
  .object({
    text: z.string().describe("Atomic user memory as a short sentence"),
    is_new: z
      .boolean()
      .describe("True if this memory is NEW. False if duplicate."),
  })
  .describe("An individual memory item");

const MemoryDecisionSchema = z.object({
  should_write: z.boolean().describe("Whether to store any memories"),
  memories: z.array(MemoryItemSchema),
});

export async function callOpenAIModel(
  usermsgs: BaseMessage[],
  sysmsg?: SystemMessage,
  structuredOutput?: boolean,
) {
  const mappedUserMessages = usermsgs.map((msg) => ({
    role: "user" as const,
    content: msg.content as string,
  }));

  const inputMessages: { role: "system" | "user"; content: string }[] = [];

  if (sysmsg) {
    inputMessages.push({
      role: "system" as const,
      content: sysmsg.content as string,
    });
  }

  const finalInputMessages = [...inputMessages, ...mappedUserMessages];

  if (structuredOutput) {
    const response = await openai.responses.parse({
      model: "gpt-5-nano",
      input: finalInputMessages,
      text: {
        format: zodTextFormat(MemoryDecisionSchema, "memory_extractor"),
      },
    });

    return response.output_parsed;
  } else {
    const response = await openai.chat.completions.create({
      model: "gpt-5-nano",
      messages: finalInputMessages,
    });

    return response.choices[0]?.message.content;
  }
}
```

### What's happening here:
- **Zod schemas** — Define the expected structured output shape:
  - `MemoryItemSchema`: `{ text: string, is_new: boolean }` — one atomic memory fact
  - `MemoryDecisionSchema`: `{ should_write: boolean, memories: MemoryItem[] }` — the full decision
- **`zodTextFormat()`** — OpenAI SDK helper that converts Zod schema to JSON Schema for strict mode
- **`openai.responses.parse()`** — Calls the LLM and parses the response into the Zod schema automatically
- **Why raw SDK?** — LangChain's `withStructuredOutput` would leak extraction tokens into the streamed chat response. Using a separate non-LangChain call avoids this entirely.
- **Model**: Uses `gpt-5-nano` (cheap and fast) since this is only for memory extraction, not user-facing responses

### Example output from the LLM:
```json
{
  "should_write": true,
  "memories": [
    { "text": "User's name is Rakesh", "is_new": true },
    { "text": "User loves Go programming", "is_new": true }
  ]
}
```

---

## Step 5: Modify the Graph (`app/api/chat/graph.ts`)

This is the core change. We need to:
1. Add a `memoryRememberNode` that extracts and stores memories
2. Modify `llmCall` to fetch and inject memories into the system prompt
3. Restructure the graph for parallel execution

Replace `app/api/chat/graph.ts` with:

```typescript
import {
  END,
  GraphNode,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { waitUntil } from "@vercel/functions";
import { v4 as uuidv4 } from "uuid";

import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

import { MessagesState } from "./state";
import { getDynamicModel } from "./model";
import { tools } from "./tools";
import { ingestEventToPolar } from "@/lib/polar";
import { getStore } from "@/lib/store";
import { callOpenAIModel } from "@/http/llm";
import {
  BASE_SYSTEM_PROMPT_TEMPLATE,
  REMEMBER_MEMORY_PROMPT,
} from "./prompts";

const checkpointer = PostgresSaver.fromConnString(
  process.env.DATABASE_URL!,
);
const store = await getStore();

// ─── NEW: Memory Extraction Node ────────────────────────────
// Runs in PARALLEL with llmCall. Extracts user facts and stores them.
const memoryRememberNode: GraphNode<typeof MessagesState> =
  async (state, runtime) => {
    console.log("[Memory] memoryRememberNode triggered");
    const userId = runtime.context?.userId;
    if (!userId) {
      console.error("userId required for memory extraction");
      return {};
    }

    const namespace = [userId, "memories"];

    const lastMessage = state.messages.at(-1);
    if (!lastMessage) return {};

    // Skip very short messages (e.g., "hi", "ok")
    const content =
      typeof lastMessage.content === "string"
        ? lastMessage.content
        : "";
    if (content.length < 5) {
      console.log(
        `[Memory] Skipping short message (${content.length} chars)`,
      );
      return {};
    }

    // 1. Fetch existing memories for deduplication context
    const existingItems = await store.search(namespace);
    const existingTexts = existingItems
      .map((it) => it.value.data)
      .filter(Boolean);

    const userDetailsContent =
      existingTexts.length > 0
        ? existingTexts.map((t) => `- ${t}`).join("\n")
        : "(empty)";

    // 2. Format the extraction prompt
    const sysMsg = await REMEMBER_MEMORY_PROMPT.format({
      user_details_content: userDetailsContent,
    });

    const sysmsg = new SystemMessage({ content: sysMsg });
    const usermsgs = [
      new HumanMessage(
        `USER MESSAGE:\n${lastMessage.content}`,
      ),
    ];

    // 3. Call OpenAI for structured memory extraction
    const memoryDecisionOutput = await callOpenAIModel(
      usermsgs,
      sysmsg,
      true,
    );

    if (
      !memoryDecisionOutput ||
      typeof memoryDecisionOutput === "string"
    ) {
      return {};
    }

    // 4. Store only NEW memories
    if (
      memoryDecisionOutput.should_write &&
      memoryDecisionOutput.memories
    ) {
      for (const mem of memoryDecisionOutput.memories) {
        if (mem.is_new) {
          await store.put(namespace, uuidv4(), {
            data: mem.text,
          });
        }
      }
    }

    console.log(
      `memory updated for user ${userId}: ${JSON.stringify(memoryDecisionOutput)}`,
    );

    return {};
  };

// ─── MODIFIED: LLM Call Node ────────────────────────────────
// Now fetches memories and injects them into the system prompt
const llmCall: GraphNode<typeof MessagesState> = async (
  state,
  runtime,
) => {
  const selectedModel = runtime.context?.selectedModel;
  const userId = runtime.context?.userId;

  const model = getDynamicModel(selectedModel);
  const modelWithTools = model.bindTools(tools);

  // NEW: Fetch user memories from store
  let memoriesContent = "(empty)";
  if (userId) {
    const namespace = [userId, "memories"];
    const existingItems = await store.search(namespace);
    const existingTexts = existingItems
      .map((it) => it.value.data)
      .filter(Boolean);

    if (existingTexts.length > 0) {
      memoriesContent = existingTexts
        .map((t) => `- ${t}`)
        .join("\n");
    }
  }

  // NEW: Use personalized system prompt instead of hardcoded one
  const formattedSystemPrompt =
    await BASE_SYSTEM_PROMPT_TEMPLATE.format({
      user_details_content: memoriesContent,
    });

  const response = await modelWithTools.invoke([
    new SystemMessage(formattedSystemPrompt),
    ...state.messages,
  ]);

  const usage = response.usage_metadata;

  waitUntil(
    (async () => {
      await ingestEventToPolar({
        userId,
        model: selectedModel,
        inputTokens: usage?.input_tokens || 0,
        outputTokens: usage?.output_tokens || 0,
        total_tokens: usage?.total_tokens || 0,
      });
    })(),
  );

  return {
    messages: [response],
  };
};

function shouldContinue(state: typeof MessagesState.State) {
  const lastMessage = state.messages.at(-1);

  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return "__end__";
  }

  if (lastMessage.tool_calls?.length) {
    return "tools";
  }

  return "__end__";
}

const toolNode = new ToolNode(tools);

// NEW: Fan-out router — runs both nodes in parallel from START
const routeFromStart = () => {
  return ["callLlm", "memoryRememberNode"];
};

// ─── MODIFIED: Graph Structure ──────────────────────────────
// Before: START -> callLlm -> shouldContinue -> tools/END
// After:  START -> [callLlm, memoryRememberNode] (parallel)
//         memoryRememberNode -> END
//         callLlm -> shouldContinue -> tools/END
export const agent = new StateGraph(MessagesState)
  .addNode("callLlm", llmCall)
  .addNode("tools", toolNode)
  .addNode("memoryRememberNode", memoryRememberNode)
  .addConditionalEdges(START, routeFromStart, {
    callLlm: "callLlm",
    memoryRememberNode: "memoryRememberNode",
  })
  .addEdge("memoryRememberNode", END)
  .addConditionalEdges("callLlm", shouldContinue, {
    __end__: END,
    tools: "tools",
  })
  .addEdge("tools", "callLlm")
  .compile({ checkpointer, store }); // <-- store is passed here
```

### Key changes explained:

**`memoryRememberNode`** (new node):
1. Gets the last user message
2. Skips if message is too short (< 5 chars)
3. Fetches existing memories from the vector store to give the LLM deduplication context
4. Calls `callOpenAIModel()` with structured output to extract `{ should_write, memories }`
5. Only stores memories where `is_new === true` via `store.put(namespace, uuid, { data })`
6. Returns `{}` — does NOT modify the graph state (memory is a side effect)

**`llmCall`** (modified):
- Now fetches memories via `store.search([userId, "memories"])`
- Formats them as a bulleted list and injects into `BASE_SYSTEM_PROMPT_TEMPLATE`
- Replaces the old hardcoded `"You are a helpful assistant."` prompt

**Graph structure** (modified):
- `routeFromStart` returns `["callLlm", "memoryRememberNode"]` — triggers parallel fan-out
- `addConditionalEdges(START, routeFromStart, { ... })` creates the fan-out from START
- Both nodes run simultaneously — memory extraction does NOT slow down the AI response
- `.compile({ checkpointer, store })` — the store is passed to the graph compiler

---

## Step 6: Create the Memories UI Page

### Server Component (`app/(chat)/memories/page.tsx`)

Create `app/(chat)/memories/` directory, then create `page.tsx`:

```typescript
import { Card, CardContent } from "@/components/ui/card";
import { Database } from "lucide-react";
import Records from "./records";
import { getStore } from "@/lib/store";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export interface Memory {
  id: string;
  content: string;
  createdAt: Date;
}

export default async function UserMemoriesPanel() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) {
    redirect("/auth/signin");
  }

  const store = await getStore();
  const namespaceForMemory = [session.user.id, "memories"];
  const allmemories = await store.search(namespaceForMemory);

  const newMemories: Memory[] = allmemories.map((item) => {
    return {
      id: item.key,
      content: item.value.data,
      createdAt: item.createdAt,
    };
  });

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 p-4">
      <Card className="rounded-2xl border shadow-sm">
        <CardContent className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div>
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              <h2 className="text-xl font-semibold">
                Memory Center
              </h2>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Persistent AI long-term contextual knowledge
            </p>
          </div>
        </CardContent>
      </Card>
      <Records memories={newMemories} />
    </div>
  );
}
```

### Client Component (`app/(chat)/memories/records.tsx`)

```typescript
"use client";

import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Clock, Search } from "lucide-react";
import { useState } from "react";
import { Memory } from "./page";

function Records({ memories }: { memories: Memory[] }) {
  const [search, setSearch] = useState("");

  const filtered = memories.filter((m) =>
    m.content?.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <Card className="rounded-2xl border shadow-sm">
      <CardHeader className="pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <CardTitle className="text-lg">
            Memory Records ({memories.length})
          </CardTitle>
          <CardDescription className="text-xs">
            Structured contextual entries stored by the AI
            system
          </CardDescription>
        </div>

        <div className="relative w-full md:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search memories..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 rounded-xl h-9"
          />
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <ScrollArea className="pr-4">
          <div className="divide-y">
            {filtered.length === 0 && (
              <div className="py-10 text-center text-muted-foreground text-sm">
                No matching memories found.
              </div>
            )}

            {filtered.map((memory, index) => (
              <div
                key={memory.id}
                className="py-4 px-4 hover:bg-muted/40 transition-colors">
                <div className="flex items-start justify-between gap-6">
                  <div className="space-y-1 max-w-3xl">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className="rounded-md text-xs capitalize">
                        {index}
                      </Badge>
                      <h3 className="font-medium text-md">
                        {memory.content}
                      </h3>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {memory.id}
                    </p>
                  </div>

                  <div className="flex items-center gap-1 text-xs text-muted-foreground whitespace-nowrap">
                    <Clock className="h-3 w-3" />
                    {new Date(
                      memory.createdAt,
                    ).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

export default Records;
```

---

## Step 7: Add Sidebar Navigation Link

In `components/sidebar/app-sidebar.tsx`:

1. Add `Database` to the lucide-react import:

```typescript
import {
  Search,
  LayoutGrid,
  Plus,
  Loader2,
  Database,    // <-- add this
} from "lucide-react";
```

2. Add the Memories item to the nav array (after "Images"):

```typescript
{
  title: "Memories",
  icon: Database,
  href: "/memories",
},
```

---

## Testing

1. **Start the dev server**: `pnpm dev`

2. **Test memory storage**: Send a message like:
   > "My name is Rakesh and I love Go programming"

   Check server console for:
   ```
   [Memory] memoryRememberNode triggered
   memory updated for user <userId>: {"should_write":true,"memories":[{"text":"User's name is Rakesh","is_new":true},{"text":"User loves Go programming","is_new":true}]}
   ```

3. **Test memory retrieval** (same or different thread):
   > "What is my name?"

   The AI should respond with "Rakesh" using the stored memory.

4. **Test the Memories page**: Navigate to `/memories` in your browser to see all stored memories with search functionality.

---

## File Summary

| File | Status | Purpose |
|------|--------|---------|
| `lib/store.ts` | **New** | PostgresStore with OpenAI embeddings |
| `app/api/chat/prompts.ts` | **New** | System prompt + memory extraction prompt templates |
| `http/llm.ts` | **New** | Raw OpenAI SDK for structured output extraction |
| `app/api/chat/graph.ts` | **Modified** | Added memoryRememberNode, modified llmCall, parallel graph |
| `app/(chat)/memories/page.tsx` | **New** | Memories page (server component) |
| `app/(chat)/memories/records.tsx` | **New** | Memories display (client component) |
| `components/sidebar/app-sidebar.tsx` | **Modified** | Added Memories nav link |

---

## Common Issues

### "Memories not being stored"
- Check server console for `[Memory] memoryRememberNode triggered` log
- If you see `[Memory] Skipping short message` — your message was too short (< 5 chars)
- If you see no memory logs at all — the node may not be running. Verify the graph structure.

### "AI doesn't use stored memories"
- Memories are injected into the system prompt in `llmCall`. Check that `store.search()` returns results.
- Visit `/memories` to confirm memories exist in the store.

### "Structured output leaking into chat stream"
- Make sure you're using the raw OpenAI SDK (`http/llm.ts`) and NOT LangChain's `withStructuredOutput` for memory extraction. LangChain would stream the structured output tokens into the user-facing response.

### "Database table errors"
- `getStore()` calls `store.setup()` once, which creates the required PostgreSQL tables. Make sure your database user has CREATE TABLE permissions.
