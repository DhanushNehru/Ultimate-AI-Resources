/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#7c3aed",
        secondary: "#06b6d4",
        accent: "#f43f5e",
        "text-main": "#f8fafc",
        "text-muted": "#94a3b8",
      },
    },
  },
  plugins: [],
}

