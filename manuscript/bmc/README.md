# BMC Medical Informatics and Decision Making version

`main_bmc.tex` uses the Springer Nature `sn-jnl` class (LPPL, December 2024 template package) with the numbered Vancouver reference style (`sn-vancouver-num`).
Structure: structured abstract, Background, Methods, Results, Discussion, Conclusions, list of abbreviations, Declarations.

Build: `cd manuscript/bmc && pdflatex main_bmc && bibtex main_bmc && pdflatex main_bmc && pdflatex main_bmc`.
Figures are read from `manuscript/figures/` and references from `manuscript/references.bib`.
The text is generated from `manuscript/main.tex`; keep the two in sync when results change.
Check the journal's current instructions for authors (abstract limit, declarations wording, figure files) before submission.
