import { HTMLAttributes } from "react";
import { cn } from "../../lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-slate-200 bg-white shadow-panel dark:border-slate-800 dark:bg-navy-900",
        className,
      )}
      {...props}
    />
  );
}
