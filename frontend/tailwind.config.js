/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,jsx}'],
  theme: {
    extend: {},
  },
  plugins: [
    require('tailwindcss/plugin')(({ addVariant }) =>
      addVariant('rtl', '[dir="rtl"] &')
    ),
  ],
}

