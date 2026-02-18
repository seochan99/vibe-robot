import type { AIModel } from "@/lib/api";

export type SimTool = "camera" | "move" | "add";

export const FALLBACK_MODELS: AIModel[] = [
  { id: "rule_based", name: "Rule-Based (No AI)", requires_auth: false },
];

export const CHATGPT_MODELS: AIModel[] = [
  { id: "rule_based", name: "Rule-Based (No AI)", requires_auth: false },
  { id: "gpt-4o", name: "GPT-4o", requires_auth: true },
  { id: "o3", name: "o3", requires_auth: true },
  { id: "o4-mini", name: "o4-mini", requires_auth: true },
];

export const SCENES = [
  { id: "wind_paper", label: "Wind & Papers", desc: "Papers blowing on a desk with a book nearby." },
  { id: "pick_and_place", label: "Pick & Place", desc: "Move the red cube to the blue bin." },
  { id: "sorting", label: "Fruit Sorting", desc: "Sort three fruits into a container." },
];

export const OBJECT_PALETTE = [
  { type: "book", label: "Book", icon: "\uD83D\uDCD6" },
  { type: "cup", label: "Cup", icon: "\u2615" },
  { type: "pen", label: "Pen", icon: "\uD83D\uDD8A" },
  { type: "paper", label: "Paper", icon: "\uD83D\uDCC4" },
  { type: "apple", label: "Apple", icon: "\uD83C\uDF4E" },
  { type: "banana", label: "Banana", icon: "\uD83C\uDF4C" },
  { type: "orange", label: "Orange", icon: "\uD83C\uDF4A" },
  { type: "cube", label: "Cube", icon: "\uD83D\uDFE5" },
];
