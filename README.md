# QA System – Question Answering on Technical Documentation

Retriever + Reader pipeline (SQuAD, FAISS, Streamlit).

## Installation
pip install -r requirements.txt

## Run
streamlit run app.py

## Project structure
- build_corpus.py: builds the corpus
- embed.py: computes embeddings / FAISS index
- reader.py: answer extraction
- eval_retrieval.py, eval_end2end.py: evaluation

  ## Presentation
[View the presentation (PDF)](My%20Presentation.pdf)
