#!/bin/sh
# compile the manuscript (run from manuscript/)
pdflatex -interaction=nonstopmode selected_responses_v11.tex > /dev/null
bibtex selected_responses_v11 > /dev/null
pdflatex -interaction=nonstopmode selected_responses_v11.tex > /dev/null
pdflatex -interaction=nonstopmode selected_responses_v11.tex > /dev/null
grep -E "^!|undefined|Overfull|Citation" selected_responses_v11.log | sort | uniq -c
