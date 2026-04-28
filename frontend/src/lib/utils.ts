import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function scoreToPercent(score?: number) {
  if (score === undefined || Number.isNaN(score)) return 0;
  return Math.max(0, Math.min(100, Math.round(score * 100)));
}

export function compactText(text: string, length = 180) {
  return text.length > length ? `${text.slice(0, length).trim()}...` : text;
}
