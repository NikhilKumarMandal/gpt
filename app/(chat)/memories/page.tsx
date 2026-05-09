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
