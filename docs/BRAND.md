# PresenceAI brand

Owner's logo files (2026-11-25), in `src/presenceai/ui/static/brand/`:

| File | Use |
|---|---|
| `presenceai-logo.svg` | logo for light backgrounds (app header, README in light mode) |
| `presenceai-logo-dark.svg` | logo for dark backgrounds (Gradio dark theme, README in dark mode) |
| `presenceai-icon.svg` | the mark alone |
| `favicon.ico` | browser tab icon (16, 32 and 48 px) |

The app serves them from its own folder (`/brand/…`); nothing is loaded from outside.

## Colours

| Name | Hex | Use |
|---|---|---|
| Navy | `#142B43` | text, the wordmark |
| Orange | `#FF7A00` | accents only |
| Orange highlight | `#FF9A32` | accents only (lighter orange) |
| Background | `#FFFFFF` | page background |

**Orange is for accents only, never as the background of white text.** WCAG contrast (computed
with the WCAG 2.x relative-luminance formula):

| Pair | Ratio | WCAG AA (4.5:1 normal text) |
|---|---|---|
| White on orange `#FF7A00` | 2.61:1 | fails |
| White on orange highlight `#FF9A32` | 2.12:1 | fails |
| Navy on white | 14.42:1 | passes |
| Navy on orange `#FF7A00` | 5.52:1 | passes |
| Navy on orange highlight `#FF9A32` | 6.81:1 | passes |

If text must sit on orange, make it navy.
