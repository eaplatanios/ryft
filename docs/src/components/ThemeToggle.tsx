import { createSignal, onMount } from 'solid-js';

type Theme = 'light' | 'dark';

/**
 * Tiny client-side toggle that flips `html[data-theme]` between light/dark and
 * persists the choice to localStorage. The initial theme is set synchronously
 * by an inline script in {@link Layout.astro} so this component can simply
 * sync with whatever's already on the element when it mounts.
 */
export default function ThemeToggle() {
  const [theme, setTheme] = createSignal<Theme>('light');

  onMount(() => {
    const current = (document.documentElement.dataset.theme as Theme) || 'light';
    setTheme(current);
  });

  function toggle() {
    const next: Theme = theme() === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('ryft-theme', next);
    } catch (_) {
      /* ignore — private mode, quota, etc. */
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle color theme"
      class="inline-flex items-center justify-center w-9 h-9 rounded-md border border-transparent hover:bg-[color-mix(in_srgb,var(--color-ryft-fg)_8%,transparent)] transition"
    >
      {/* Moon — shown in light mode (click to switch to dark) */}
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        width="18"
        height="18"
        class="block dark:hidden"
      >
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
      </svg>
      {/* Sun — shown in dark mode (click to switch to light) */}
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        width="18"
        height="18"
        class="hidden dark:block"
      >
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
      </svg>
    </button>
  );
}
