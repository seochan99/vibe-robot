# ArmCraft CHI 2027 Draft

This folder contains an ACM `acmart`-based draft for the ArmCraft CHI 2027 submission concept.

## Contents

- `main.tex` — paper draft
- `references.bib` — bibliography
- `proposal-update.md` — updated proposal summary (Korean)
- `claim_evidence_matrix.md` — claim-to-evidence traceability map (literature + artifact)
- `figures/` — generated figures used in the paper
  - `framework_tikz.tex` (LaTeX-native full framework figure)
  - `study_design_tikz.tex` (LaTeX-native Study 1/2 evaluation design figure)
  - `0.onboarding-screen.png` to `5.approve-and-work.png` (real UI screenshot sequence for runtime walkthrough)
  - `failure_inspector.png`, `interaction_loop.png` (supporting conceptual figures)
  - `system_overview.png`
  - `study_design.png` (legacy draft asset, not used in current `main.tex`)
- `scripts/make_figures.py` — figure generation script
- `acmart.cls`, `ACM-Reference-Format.bst` — copied from local `acmart-primary` template

## Build

```bash
cd /Users/chan/viberobot/paper/chi2027
latexmk -pdf main.tex
```

Output: `main.pdf`

## Notes

- This draft is written as a CHI 2027 proposal-style full paper with planned evaluations (not fabricated user-study results).
- Replace metadata placeholders (`DOI`, `ISBN`, conference rights details) at submission time.
- Update/verify bibliography metadata before camera-ready.
