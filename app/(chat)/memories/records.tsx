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
