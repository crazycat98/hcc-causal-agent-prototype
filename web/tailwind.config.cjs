/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        panel: "0 1px 2px rgb(15 23 42 / 0.06)",
      },
    },
  },
  plugins: [],
};
