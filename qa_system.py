"""
Le systeme complet (retriever + reader + seuil), utilisable depuis n'importe quelle interface.

    from qa_system import QASystem
    qa = QASystem(method="bm25", k=5)
    result = qa.ask("What does git gc do?")
"""
import html
import re
from pathlib import Path

from common import load_json
from reader import DEFAULT_READER, Reader
from retrievers import build_retriever

DEFAULT_THRESHOLD = 0.01


def doc_url(source: str) -> str:
    """git_scm_com_docs_git_format_patch.txt -> https://git-scm.com/docs/git-format-patch"""
    name = re.sub(r"^git_scm_com_docs_", "", Path(source).stem).replace("_", "-")
    return f"https://git-scm.com/docs/{name}"


def highlight(text: str, answer: str) -> str:
    """Passage en HTML, avec la reponse surlignee (premiere occurrence)."""
    i = text.find(answer) if answer else -1
    if i == -1:
        return html.escape(text)
    return html.escape(text[:i]) + "<mark>" + html.escape(answer) + "</mark>" + html.escape(text[i + len(answer):])


class QASystem:
    def __init__(self, method="bm25", k=5, reader_model=DEFAULT_READER, corpus_path="corpus.json"):
        self.corpus = load_json(corpus_path)
        self.by_id = {p["id"]: p for p in self.corpus}
        self.retriever = build_retriever(method, self.corpus)
        self.reader = Reader(reader_model)
        self.k = k
        self.threshold = DEFAULT_THRESHOLD
        if Path("threshold.json").exists():                 # seuil choisi par tune_threshold.py
            self.threshold = load_json("threshold.json").get("min_score", DEFAULT_THRESHOLD)

    def ask(self, question, k=None, min_score=None):
        k = k or self.k
        min_score = self.threshold if min_score is None else min_score
        passages = [self.by_id[self.retriever.ids[i]] for i in self.retriever.search(question, k)]
        best = self.reader.answer(question, passages, 0.0, allow_abstain=False)
        answered = bool(best["answer"]) and best["score"] >= min_score
        source = self.by_id.get(best["passage_id"])
        return {
            "question": question,
            "answered": answered,                           # False -> afficher "je ne sais pas"
            "answer": best["answer"] if answered else "",
            "best_guess": best["answer"],                   # meilleure hypothese, meme non fiable
            "score": best["score"],
            "threshold": min_score,
            "passage": source,                              # {"id","source","text"} ou None
            "retrieved": passages,
        }