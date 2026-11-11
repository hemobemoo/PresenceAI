---
description: Walk through the running Gradio app with Playwright and report problems
---

Start the app with `uv run presenceai` in the background and wait until http://127.0.0.1:7860 responds. Using the Playwright MCP server:

1. Open the app; switch language EN → AR → EN; screenshot each.
2. Go through consent → goal → practice using the fixture upload path (not the live camera) → report → progress → data.
3. On each page check: no untranslated keys, RTL layout in Arabic, no emotion/personality words, every number on the report has a confidence label.

Save screenshots to `experiments/results/ui-check/` and list anything broken. Stop the app at the end. Do not change code unless I ask.
