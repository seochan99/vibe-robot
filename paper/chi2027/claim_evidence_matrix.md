# ArmCraft CHI2027 Claim-Evidence Matrix (v2)

Date: 2026-02-19  
Scope: Map major claims in `/Users/chan/viberobot/paper/chi2027/main.tex` to verifiable evidence.

## A. Literature-Backed Claims

| Claim ID | Claim (short) | Evidence type | Source |
|---|---|---|---|
| L1 | LLM-based robot planning/control has rapidly advanced | Peer-reviewed / preprint robotics literature | SayCan, Inner Monologue, ProgPrompt, Code as Policies, VoxPoser, PaLM-E, RT-2 (`ahn2022saycan`, `huang2022innermonologue`, `singh2022progprompt`, `liang2023codeaspolicies`, `huang2023voxposer`, `driess2023palme`, `brohan2023rt2`) |
| L2 | LLM-mediated robot EUD is emerging | HRI + arXiv | Alchemist (HRI 2024), Cocobo (`karli2024alchemist`, `ge2024cocobo`) |
| L3 | Explanations should support action and mental-model repair | HCI/IUI evidence | `kulesza2015principles`, `kulesza2013too`, `liao2020questioning`, `abdul2018trends` |
| L4 | Trust in automation should be calibrated, not maximized | Human factors/HRI evidence | `lee2004trust`, `parasuraman1997humans`, `hancock2011meta` |
| L5 | NASA-TLX is a standard workload measure | Measurement literature | `hart1988nasa` |
| L6 | Mixed-effects modeling is appropriate for repeated-measures confirmatory analysis | Methodology literature | `barr2013maximal` |
| L7 | Report effect sizes + CI, not only p-values | Methodology literature | `lakens2013effectsizes` |
| L8 | Thematic analysis is valid for qualitative coding | Qualitative methods literature | `braun2006thematic` |
| L9 | RT-2 reports large-scale evaluation over 6k trials | Quantitative result in abstract | `brohan2023rt2` |
| L10 | PaLM-E reports a 562B-parameter embodied multimodal model | Quantitative model-scale report in abstract | `driess2023palme` |
| L11 | Code as Policies reports 39.8% HumanEval | Quantitative benchmark report in abstract | `liang2023codeaspolicies` |
| L12 | HRI trust meta-analysis reports 29 studies with aggregate effects $\bar{r}=0.26$, $\bar{d}=0.71$ | Quantitative meta-analysis result | `hancock2011meta` |

## B. Artifact-Backed System Claims

| Claim ID | Claim (short) | Evidence type | Artifact evidence |
|---|---|---|---|
| S1 | Pipeline enforces sim-first preview before approval | Code path | `/Users/chan/viberobot/core/pipeline.py:446`, `/Users/chan/viberobot/core/pipeline.py:469` |
| S2 | Plan pauses at `awaiting_approval` when auto-approve is off | Code + test | `/Users/chan/viberobot/core/pipeline.py:471`, `/Users/chan/viberobot/tests/test_pipeline.py:61` |
| S3 | Execution only starts after explicit approval | Code + test | `/Users/chan/viberobot/core/pipeline.py:744`, `/Users/chan/viberobot/tests/test_pipeline.py:75` |
| S4 | Live panel is frozen during preview | Code comment + lock | `/Users/chan/viberobot/core/pipeline.py:788` |
| S5 | Retry target lock guardrail exists | Code + tests | `/Users/chan/viberobot/core/pipeline.py:982`, `/Users/chan/viberobot/tests/test_pipeline.py:90`, `/Users/chan/viberobot/tests/test_retry_context.py:40` |
| S6 | Context resolver fallback avoids hard-coded reference keywords | Code | `/Users/chan/viberobot/api/context_resolver.py:153` |
| S7 | Failure Inspector cards map failures to actionable fixes | Code | `/Users/chan/viberobot/ui/failure_cards.py:40`, `/Users/chan/viberobot/ui/failure_cards.py:133` |
| S8 | Safety contracts include preconditions/invariants/tripwires | Code + test | `/Users/chan/viberobot/core/safety_contract.py:79`, `/Users/chan/viberobot/tests/test_pipeline.py:127` |
| S9 | Scene re-init removes runtime-added objects (isolation) | API test | `/Users/chan/viberobot/tests/test_api_scene_isolation.py:13` |
| S10 | Study logging captures commands/approval/failure/patch/timing | Code | `/Users/chan/viberobot/study/logger.py:66`, `/Users/chan/viberobot/study/logger.py:93`, `/Users/chan/viberobot/study/logger.py:104`, `/Users/chan/viberobot/study/logger.py:108`, `/Users/chan/viberobot/study/logger.py:112` |
| S11 | Framework figure is LaTeX-native (not raster-only), enabling source-level auditability | TeX artifact | `/Users/chan/viberobot/paper/chi2027/figures/framework_tikz.tex:1` |

## C. Local Regression Execution Record

Command executed:

```bash
cd /Users/chan/viberobot && python3 -m venv .venv && . .venv/bin/activate && python -m pip install -q pytest && python -m pytest -q tests/test_pipeline.py::TestPipelineE2E::test_pipeline_awaiting_approval tests/test_pipeline.py::TestPipelineE2E::test_approve_and_execute tests/test_pipeline.py::TestPipelineE2E::test_retry_guardrail_keeps_preferred_target tests/test_retry_context.py::test_context_fallback_locks_previous_target_after_failure tests/test_api_scene_isolation.py::test_scene_init_resets_runtime_added_objects
```

Result:

- `5 passed in 11.08s`

Interpretation:

- The selected regression set supports claims S1-S6 and S9 at test level.
- This does **not** constitute Study 1/2 user-study evidence; efficacy claims remain hypotheses in `main.tex`.
