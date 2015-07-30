all:
	pdflatex -interaction=nonstopmode -halt-on-error paper
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper
anon:
	pdflatex -interaction=nonstopmode -halt-on-error paper_anon
	bibtex paper_anon
	pdflatex -interaction=nonstopmode -halt-on-error paper_anon
	bibtex paper_anon
	pdflatex -interaction=nonstopmode -halt-on-error paper_anon
	bibtex paper_anon
	pdflatex -interaction=nonstopmode -halt-on-error paper_anon
spellcheck:
	cat paper.tex | aspell pipe check --mode=tex --personal=./words.en.pws | grep -v -e  '^*$$' | grep -e '^$$' -v
