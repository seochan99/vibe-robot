LATEXMK=latexmk
TARGET=main

all: pdf

pdf:
	$(LATEXMK) -pdf $(TARGET).tex

clean:
	$(LATEXMK) -C
	rm -f *.bbl *.blg *.run.xml *.synctex.gz

figures:
	python3 scripts/make_figures.py
