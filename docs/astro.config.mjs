import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import solid from '@astrojs/solid-js';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://ryft.dev',
  // The Astro site lives at `/`; the mdBook user guide is dropped into `/book/`
  // by the CI workflow after both builds finish.
  integrations: [
    solid(),
    mdx(),
    sitemap(),
  ],
  vite: {
    plugins: [tailwindcss()],
    // Allow Vite to read files from the workspace root so MDX pages can embed
    // Rust source from `crates/ryft/examples/*.rs` via the `?raw` import.
    server: { fs: { allow: ['..'] } },
  },
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      defaultColor: false,
      wrap: false,
    },
  },
});
