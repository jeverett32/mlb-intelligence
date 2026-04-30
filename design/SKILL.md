---
name: mlb-intelligence-design
description: Use this skill to generate well-branded interfaces and assets for MLB Intelligence (jeverett32/mlb-pipeline), either for production or throwaway prototypes/mocks/etc. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out and create static HTML files for the user to view. The canonical token file is `colors_and_type.css` — link it directly. Brand mark is `assets/logo.svg`. MLB team logos live in `assets/team-logos/<ABBR>.png`.

If working on production code, you can copy assets and read the rules here to become an expert in designing with this brand. Match the dual-surface convention: glass + orbits + Playfair display only on the public marketing surface (`landing.html`, `public.html`); flat warm-paper cards everywhere else (operator dashboard, admin).

If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.
