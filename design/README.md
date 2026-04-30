# MLB Intelligence — Design System

A working design system reverse-engineered from the production
[`jeverett32/mlb-pipeline`](https://github.com/jeverett32/mlb-pipeline) repo:
a Python/FastAPI dashboard that fronts an ML pipeline predicting MLB games
and executing wagers on [Kalshi](https://kalshi.com).

The product positions itself as **MLB Intelligence** — "a public audit trail
for a model-driven baseball betting workflow."

## Sources used to build this system

All real, all from the codebase — no Figma, no screenshots, no guesswork.

| Source | Path in repo |
|---|---|
| Public-facing CSS (~1500 lines) | `dashboard/static/css/public-site.css` |
| Private dashboard primitives | `dashboard/static/css/dashboard.css` |
| Marketing landing template | `dashboard/templates/landing.html` |
| Public analytics page | `dashboard/templates/public.html` |
| Operator dashboard | `dashboard/templates/index.html` |
| Admin console | `dashboard/templates/admin.html` |
| Login / register / pending | `dashboard/templates/{login,register,pending}.html` |
| Settings / contact / privacy / api-docs | `dashboard/templates/*.html` |
| Brand mark | `dashboard/static/favicon.svg` (re-saved as `assets/logo.svg`) |
| Webfonts | `dashboard/static/fonts/*.woff2` |
| MLB team logos | `dashboard/static/team-logos/*.png` (30 teams) |

## The three surfaces

The product has three distinct audiences and the design system treats them
that way:

1. **Public marketing + analytics** — `/`, `/public`, `/contact`, `/privacy`,
   `/login`, `/register`. Editorial layout, warm off-white surface, gold
   accent. The hero is the "track record"; the headline is "Public MLB
   model results, updated daily." Calibration plots and ROI curves live here.
2. **User dashboard** — `/dashboard` (approved users only). Operator view of
   live betting state, exposure, bankroll, recent bets, model output. Same
   palette, denser layout, more tables and Chart.js panels.
3. **Admin console** — `/admin`. Database browsing, user approval queue,
   deeper model insights, server-side error log. Same primitives, more
   utilitarian.

## Index — what's in this folder

| File / folder | What it is |
|---|---|
| `README.md` | This file. Start here. |
| `SKILL.md` | Agent-skill manifest so this can be loaded in Claude Code. |
| `colors_and_type.css` | All design tokens — colors, type scale, radii, shadows, spacing. |
| `fonts/` | DM Sans + Playfair Display, both woff2, both lifted from prod. |
| `assets/logo.svg` | The "diamond + curves" brand mark (was `favicon.svg`). |
| `assets/team-logos/` | All 30 MLB team logos as PNG, keyed by abbreviation. |
| `preview/` | Small HTML cards that populate the Design System tab. |
| `ui_kits/public-site/` | Hi-fi recreation of the public marketing + analytics surface. |
| `ui_kits/user-dashboard/` | Hi-fi recreation of the approved-user operator view. |
| `ui_kits/admin/` | Hi-fi recreation of the admin console. |

## Content fundamentals

The voice is **measured, evidentiary, mildly editorial**. It reads like a
quant fund's investor letter, not a sports-pick Twitter account. Specifics:

- **Person.** Third person about the system ("the model", "the pipeline"),
  second person sparingly to address the visitor ("Visitors can explore
  the analytics layer first"). Almost never first person.
- **Tone.** Dry, evidence-first, slightly skeptical of itself. The landing
  copy literally says: *"The point of this site is not to spray picks. It
  is to make the workflow legible."*
- **Casing.** Sentence case for headings and CTAs ("Request Access",
  "View Public Analytics", "Public metrics, not marketing claims.").
  Title Case is reserved for proper nouns. Eyebrows are ALL CAPS with
  +0.14em tracking.
- **Numbers.** Tabular figures (`font-feature-settings: 'tnum'`),
  percentages always one decimal (`52.4%`), signed where direction matters
  (`+3.2%`, `−1.8%`). No K/M abbreviations on the public surface — full
  numbers ("Total bets: 1,247").
- **Hedging language.** Heavy use of qualifiers: "tracked", "settled",
  "audit trail", "evidence layer", "receipts". The word "guarantee" never
  appears. "Edge" is used precisely (model probability minus market
  probability, in points).
- **Eyebrows as section labels.** Every section opens with a small all-caps
  eyebrow ("PROCESS", "MODEL INSIGHTS", "RECEIPTS", "TRACK RECORD") above
  a Playfair display heading.
- **Emoji.** None. Not on the public surface, not in dashboards, not in
  errors. Don't add them.
- **Iconography in copy.** Sparse Unicode arrows and bullet points only —
  the proof-strip uses a literal `>` character as its chevron, not an
  icon font.
- **Microcopy examples to mimic:**
  - Error: "Could not load public analytics. Refresh and try again."
  - Empty: "No settled public bets yet."
  - Section header: "Public metrics, not marketing claims."
  - Tag chip: "Public proof" / "Private operator controls" / "Live model metrics"

## Visual foundations

### Palette
Warm-paper light theme is the canonical surface. Background `#faf9f7`,
ink `#1a1814`. The accent is **dark goldenrod `#b8860b`** — a
deliberately old-school, almanac-y gold rather than a saturated CTA color.
Wins are forest green `#2d6a4f`; losses are deep crimson `#9b2335`.
Both feel printed, not screen-bright.

A near-identical dark theme exists with `#12171c` background and
`#e0a446` (warmer, lighter gold) accent. Both ship; users pick via a
top-bar `<select>`.

### Type
Two faces. **Playfair Display 600/700** for h1/h2 and brand wordmark — it
gives the site its "Bloomberg-meets-Bookforum" editorial register.
**DM Sans 400/500/600/700** does everything else: body copy, labels,
buttons, table cells. Tabular numerals everywhere they appear.
Letter-spacing on display headings is aggressive: `-0.05em` to `-0.08em`.
Body line-height is 1.6.

### Layout & spacing
- Page shell: `max-width: 1180px`, padded `28px 32px 88px`.
- 4px base spacing scale. Most rhythm is in 16/24/28/52 increments.
- Section separation is done with **hairlines** (`1px solid var(--border)`)
  and big top/bottom margins (72px), not background blocks.
- Two-column editorial bands (`0.95fr 1.2fr`) for "lead + body" patterns.

### Backgrounds
- Pages mostly use the flat warm off-white. The landing hero adds
  **two soft radial-gradient washes** (gold from top-left, green from
  upper-right) over the base.
- Hero "demo" cards use a layered glass treatment:
  `linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.34))` +
  `backdrop-filter: blur(16px)` + a 1px white inner highlight.
- Three concentric `.orbit` rings (`spinSlow` 16-24s) drift behind the
  hero — the only purely decorative motion on the page.
- No raster background images. No textures. No grain. No patterns.

### Animation & motion
- `[data-reveal]` scroll-in: `translateY(26px) → 0` + opacity `0 → 1`
  over `0.7s cubic-bezier(0.2, 0.8, 0.2, 1)`. Universal.
- Hero floats: `floatCard` 7s ease-in-out, ±8px Y.
- Demo bars: `barPulse` 5.2s, `scaleY(0.96 ↔ 1.04)`.
- Marquee: 28s linear infinite drift.
- Orbits: 16-24s linear rotation.
- All motion respects `prefers-reduced-motion`.
- No bounce. No spring. No skeuomorphic physics. Easing is calm.

### Hover & press
- Primary buttons: `transform: translateY(-1px); filter: brightness(1.04)`.
- Cards: `translateY(-2px to -4px)` + bumped shadow.
- Links: color shift to `--accent-strong`, no underline change.
- Press: no separate state defined — the system trusts hover + browser default.

### Borders, radii, shadows
- Borders are warm gray `#e8e4de`, always 1px.
- **Radius vocabulary by context:**
  - Buttons / chips / pagination → `999px` pill
  - Inputs → `14px`
  - Standard cards / panels → `12-22px`
  - Hero glass panels → `24-32px`
  - Auth shell → `28px`
- Shadows are warm and subtle:
  - `--shadow-sm: 0 1px 3px rgba(26,24,20,0.04)` — default for cards
  - `--shadow-md: 0 4px 12px rgba(26,24,20,0.06)` — hover
  - `--shadow-lg: 0 28px 80px rgba(26,24,20,0.12)` — hero glass only
- An additional `inset 0 1px 0 rgba(255,255,255,0.46)` sits on glass
  panels for a soft top highlight.

### Transparency & blur
Used deliberately, only on the public marketing surface, only on
"glass" panels (hero demo, hero badges, marquee chips, glance cards,
analytics overview cards on `/public`). Always paired with a
white-translucent border (`rgba(255,255,255,0.34)`) and `backdrop-filter:
blur(12-16px)`. The dashboard and admin surfaces are flat — no glass.

### Color vibe of imagery
Team-logo PNGs are the only imagery. They sit on white cards, no tint,
no filter. Charts use a single accent stroke (gold) on neutral grid lines —
no gradient fills, no multi-series rainbows.

### Layout rules / fixed elements
- The `.topbar` is **not** sticky. It scrolls away with the page.
- `<a href="#main-content" class="skip-link">` is the first focusable
  element on every page.
- Mobile nav collapses into a hamburger below 560px.
- The auth shell is centered, two-column on desktop, single-column on
  mobile, hero hidden below 560px.

## Iconography

There is essentially **no icon system** in the codebase. This is intentional
and worth preserving:

- The brand mark (`assets/logo.svg`) is the only proprietary glyph: a
  gold ring with two opposing curves on a near-black square, evoking
  baseball seam stitching.
- Process steps use **numerals in a 1px ring** (`.step-number`,
  `.step-icon`) — "01", "02", "03", "04" — instead of icons.
- The proof-strip uses a literal `>` character.
- The hamburger is three CSS divs.
- The methodology cycle uses **inline SVG paths with arrowhead markers**
  drawn directly in the template — not an icon library.
- No emoji. No Lucide. No Font Awesome. No Heroicons.

**For new designs:** prefer numerals, eyebrows, and short text labels over
icons. If an icon is genuinely needed, use **Lucide** (1.5px stroke, rounded)
as the closest match in feel — and flag it as a substitution. Document any
additions here so the system stays cohesive.

## Caveats / things to verify

- The `index.html` operator dashboard and `admin.html` console are 100k+
  characters each — the UI kits cover their dominant patterns but not every
  edge-case panel.
- The dark theme is documented but the UI kits demo the light theme by
  default. Toggle via the top bar.
- Fonts ship as woff2 with a Latin subset only. Extended characters fall
  back to system stack.
