/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./web/app/**/*.{ts,tsx}", "./web/components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        panel: "0 1px 2px rgb(15 23 42 / 0.06)",
      },
    },
  },
  plugins: [],
};
