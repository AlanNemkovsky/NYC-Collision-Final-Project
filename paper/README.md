# Paper folder

This folder contains the finalized NeurIPS-style report for the project.

Important files:

- `main.tex`: LaTeX source for the final paper
- `main.pdf`: compiled PDF generated from `main.tex`
- `figures/`: figures referenced by the paper
- `tables/`: CSV tables used to support the reported metrics
- `neurips_2026.sty`: local NeurIPS-compatible style file used for compilation

To rebuild the paper from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
