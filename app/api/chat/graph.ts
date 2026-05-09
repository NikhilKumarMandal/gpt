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

    const existingItems = await store.search(namespace);
    const existingTexts = existingItems
      .map((it) => it.value.data)
      .filter(Boolean);

    const userDetailsContent =
      existingTexts.length > 0
        ? existingTexts.map((t) => `- ${t}`).join("\n")
        : "(empty)";

    const sysMsg = await REMEMBER_MEMORY_PROMPT.format({
      user_details_content: userDetailsContent,
    });

    const sysmsg = new SystemMessage({ content: sysMsg });
    const usermsgs = [
      new HumanMessage(
        `USER MESSAGE:\n${lastMessage.content}`,
      ),
    ];

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

const llmCall: GraphNode<typeof MessagesState> = async (
  state,
  runtime,
) => {
  const selectedModel = runtime.context?.selectedModel;
  const userId = runtime.context?.userId;

  const model = getDynamicModel(selectedModel);
  const modelWithTools = model.bindTools(tools);

  // Fetch user memories from store
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

const routeFromStart = () => {
  return ["callLlm", "memoryRememberNode"];
};

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
  .compile({ checkpointer, store });
