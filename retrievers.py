"""
ETAPE 4a - RETRIEVER : trouve les K passages les plus proches d'une question.

Trois methodes, meme interface  .search(question, k) -> liste d'indices dans le corpus :
    bm25    recherche lexicale (mots-cles) ; tres bon sur les noms d'options (--quiet, gc.auto...)
    dense   recherche semantique (embeddings de embed.py)
    hybrid  fusion des deux classements (Reciprocal Rank Fusion)
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from embed import DEFAULT_MODEL, encode_queries, ensure_embeddings, load_model


# ----------------------------------------------------------------------
# Tokenisation adaptee a la doc technique : "--no-walk" donne "--no-walk", "no", "walk"
# ----------------------------------------------------------------------
def tokenize(text):
    out = []
    for t in re.findall(r"-{0,2}\w[\w.\-]*", text.lower()):
        t = t.rstrip(".-")
        if not t:
            continue
        out.append(t)
        parts = [x for x in re.split(r"[.\-_]", t) if x]
        if len(parts) > 1:
            out.extend(parts)
    return out


class BM25Retriever:
    name = "bm25"

    def __init__(self, ids, texts, k1=1.5, b=0.75):
        self.ids, self.texts, self.k1, self.b = ids, texts, k1, b
        docs = [Counter(tokenize(t)) for t in texts]
        self.dl = np.array([sum(d.values()) for d in docs], dtype=float)
        self.avg = self.dl.mean()
        self.inv = defaultdict(list)                       # terme -> [(doc, frequence)]
        for i, d in enumerate(docs):
            for term, tf in d.items():
                self.inv[term].append((i, tf))
        n = len(texts)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.inv.items()}

    def scores(self, query):
        s = np.zeros(len(self.ids))
        for term in set(tokenize(query)):
            for i, tf in self.inv.get(term, ()):
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s

    def search(self, query, k=10):
        return list(np.argsort(-self.scores(query))[:k])


class DenseRetriever:
    name = "dense"

    def __init__(self, corpus, model_name=DEFAULT_MODEL):
        self.ids = [p["id"] for p in corpus]
        self.texts = [p["text"] for p in corpus]
        self.model_name = model_name
        self.model = load_model(model_name)
        self.emb = ensure_embeddings(model_name, corpus, model=self.model)   # recharge le .npy s'il existe

    def scores(self, query):
        q = encode_queries(self.model, self.model_name, [query])[0]
        return self.emb @ q                                  # vecteurs normalises -> cosinus

    def search(self, query, k=10):
        return list(np.argsort(-self.scores(query))[:k])


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25, dense, rrf_k=60, depth=100):
        self.bm25, self.dense, self.rrf_k, self.depth = bm25, dense, rrf_k, depth
        self.ids, self.texts = bm25.ids, bm25.texts

    def search(self, query, k=10):
        fused = defaultdict(float)
        for r in (self.bm25, self.dense):
            for rank, i in enumerate(r.search(query, self.depth)):
                fused[i] += 1.0 / (self.rrf_k + rank + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def build_retriever(method, corpus, model_name=DEFAULT_MODEL):
    ids, texts = [p["id"] for p in corpus], [p["text"] for p in corpus]
    if method == "bm25":
        return BM25Retriever(ids, texts)
    dense = DenseRetriever(corpus, model_name)
    if method == "dense":
        return dense
    return HybridRetriever(BM25Retriever(ids, texts), dense)