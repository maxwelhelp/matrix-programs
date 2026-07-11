#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
elif command -v bibtex.original >/dev/null 2>&1; then
  bibtex.original main
elif command -v bibtex8 >/dev/null 2>&1; then
  bibtex8 main
else
  echo "No BibTeX executable found" >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
