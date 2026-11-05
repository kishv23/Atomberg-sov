.PHONY: install install-full test analyze run dashboard

install:        ## core + agent + tests (lexicon NLP backends)
	pip install -e ".[dev,agent,app]"

install-full:   ## adds YouTube client, transformers and PyABSA
	pip install -e ".[dev,agent,app,youtube,nlp,absa]"

test:
	pytest -q

analyze:        ## score the CSVs already in data/
	sov analyze

run:            ## crawl with the agent (needs API keys) and analyze
	sov run

dashboard:
	streamlit run src/app.py
