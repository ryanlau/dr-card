/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/layout.tsx",
    "./app/page.tsx",
    "*.{js,ts,jsx,tsx,mdx}",
    "app/**/*.{ts,tsx}",
    "components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-public-sans)'],
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        ripple1: {
          "0%": {
            strokeDashoffset: "1000",
            opacity: "1",
          },
          "100%": {
            strokeDashoffset: "0",
            opacity: "0",
          },
        },
        ripple2: {
          "0%": {
            strokeDashoffset: "1000",
            opacity: "0.7",
          },
          "100%": {
            strokeDashoffset: "0",
            opacity: "0",
          },
        },
        ripple3: {
          "0%": {
            strokeDashoffset: "1000",
            opacity: "0.4",
          },
          "100%": {
            strokeDashoffset: "0",
            opacity: "0",
          },
        },
      },
      animation: {
        "ripple-1": "ripple1 3s ease-out infinite",
        "ripple-2": "ripple2 3s ease-out infinite 1s",
        "ripple-3": "ripple3 3s ease-out infinite 2s",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}

