"""The Gradio app (P3). Importing this package turns Gradio's analytics off **before** Gradio is
imported: `import gradio` otherwise starts a thread that calls api.gradio.app, and Blocks start a
version check (owner, P3 plan: no outside connection; tests/safety/test_app_network.py)."""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
