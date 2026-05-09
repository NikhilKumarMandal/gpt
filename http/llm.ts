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
