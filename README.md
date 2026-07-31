# Amortized Data Borrowing with Exchangeability-Aware Neural Posterior Estimation

This repository contains the camera-ready source files and supporting code for the MLHC 2026 paper:

**Amortized Data Borrowing with Exchangeability-Aware Neural Posterior Estimation**

## Paper source

The main manuscript is compiled from `main.tex`, with section files in `sections/` and references in `refs.bib`. The paper uses the MLHC/PMLR JMLR template files included in this repository.

To compile the paper, use a standard LaTeX workflow such as:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Code and results

The repository includes code for the simulation study, ADNI analysis pipeline, runtime comparison, and camera-ready robustness checks. ADNI raw data are not redistributed here because access requires a separate ADNI data-use agreement.
