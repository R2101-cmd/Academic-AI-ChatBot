import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#07111f",
          900: "#0a1728",
          800: "#10243d",
          700: "#16304f",
        },
        slateAcademic: "#56616f",
        silver: "#d7dde6",
        cyanSoft: "#38bdf8",
      },
      boxShadow: {
        academic: "0 18px 50px rgba(7, 17, 31, 0.12)",
        panel: "0 10px 28px rgba(7, 17, 31, 0.09)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
} satisfies Config;
