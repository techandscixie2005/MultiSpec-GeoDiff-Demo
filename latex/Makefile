PDF=main.pdf
LATEXMK=latexmk
LATEXFLAGS=-xelatex -interaction=nonstopmode -halt-on-error

all: $(PDF)

$(PDF): main.tex refs.bib sections/*.tex tables/*.tex figures/*.tex
	$(LATEXMK) $(LATEXFLAGS) main.tex

clean:
	$(LATEXMK) -C
	rm -f *.bbl *.run.xml *.bcf *.blg *.nav *.snm *.vrb

.PHONY: all clean
