# S4 terms review: training on outputs (T2 teacher) and prompt use

Read on 2026-11-13 from the providers' own pages (URLs below). The data-policy flags (trains /
retains / requires user ids) come from OpenRouter's model pages and are recorded by
`policies.py` in `experiments/results/s4/20261113-143154_policies.csv`. This file adds what the
licence and the provider's terms say. Not legal advice. Rule (owner, 2026-11-13): if training on
outputs cannot be confirmed, the verdict is **unknown**, never "allowed".

| Model | Weights licence | Licence on outputs | Serving provider's terms on using outputs to train other models | T2 teacher verdict |
|---|---|---|---|---|
| `nvidia/nemotron-3.5-super-vl:free`, `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenMDW-1.1 | No restriction ("does not impose any restrictions or obligations with respect to any use … of any outputs") | NVIDIA API Trial ToS §1.2: access "for limited trial purposes only and without use of the API Service or Generated Content in production"; §4.12: no use of Generated Content "to develop or improve products or services that compete with the API Service" | **unknown** (trial-only terms) |
| `google/gemma-4-31b-it:free` | Apache-2.0 | Not restricted | Gemini API Additional Terms (Google AI Studio), Use Restrictions: "You may not use the Services to develop models that compete with the Services" | **unknown** (depends on "compete") |
| `thinkingmachines/inkling-2-small:free` | Apache-2.0 | Not restricted | TML free research API tier ToS: may not "develop, train, fine-tune, distill, or materially improve a competing model or service" without consent | **unknown** (depends on "competing") |
| `recursive/flint-v1.0:free` | Apache-2.0 | Not restricted | Recursive model terms: no restriction on outputs found; user owns Outputs (§5.8) | **unknown** (not stated) |
| `dots-studio/dots-3-note-preview:free` | Apache-2.0 | Not restricted | AtlasCloud Acceptable Use §7: no use "to build a competitive product"; no output-training clause | **unknown** (not stated) |
| `liquid/lfm-3-1.7b:free` | LFM Open License 1.0 (commercial use only below USD 10 M revenue) | Not stated | Liquid ToS §7: may not "use the Services or outputs to develop, train, fine-tune, or improve other AI models" | **prohibited** |

## Prompt use stated in the terms (for consent wording)

- **NVIDIA** (Trial ToS §3.3): collects "User Content and Generated Content to improve NVIDIA
  products and services, including AI models"; use is logged for security and abuse monitoring.
- **Google AI Studio** (Gemini API Additional Terms, Unpaid Services): submitted content and
  responses are used to improve Google products; human reviewers may read them after they are
  disconnected from the account. OpenRouter lists 55-day retention.
- **Thinking Machines** (free research tier ToS): perpetual licence to retain and use prompts and
  outputs, "including to train, fine-tune, evaluate, and improve the Company Models".
- **Recursive** (model terms §5.2(c), §5.3, §5.9): licence to "train, fine-tune, evaluate … our
  models" on customer data, retained "including in training datasets"; it honours platform
  opt-out signals. **This conflicts with OpenRouter's flag `training: false` for this endpoint**,
  so the consent wording must not rely on that flag.
- **AtlasCloud** (data-retention §5): does not train Atlas-hosted models on customer content
  unless authorised; third-party model providers' own terms may apply.
- **Liquid** (ToS §6): may use content "to develop and improve Liquid AI's models".
- **OpenRouter** (data-collection guide; ToS §6.5): does not store prompts or responses unless the
  account opts in to logging. What OpenRouter sends to providers that "require user IDs" is not
  stated in the pages read; the app sends no `user` field.

## Sources

- https://openmdw.ai/license/1-1/
- https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf
- https://ai.google.dev/gemma/docs/gemma_4_license ; https://aistudio.google.com/docs/terms
- https://thinkingmachines.ai/legal/tml-free-research-api-tier-terms-of-service.pdf
- https://recursive.com/legal/model-terms ; https://recursive.com/legal/terms-of-use
- https://www.atlascloud.ai/privacy ; https://www.atlascloud.ai/data-retention ; https://www.atlascloud.ai/acceptable-use
- https://www.liquid.ai/terms-conditions ; https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE
- https://openrouter.ai/terms ; https://openrouter.ai/docs/guides/privacy/data-collection
