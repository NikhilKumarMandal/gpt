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
