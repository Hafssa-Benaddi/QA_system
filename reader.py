"""
ETAPE 4b - READER : extrait la reponse (un morceau de texte) d'un passage.

Modele : deepset/roberta-base-squad2 (deja entraine sur SQuAD 2.0, sait dire "pas de reponse").
Install : pip install transformers torch

Version sans "pipeline" : on charge le modele et le tokenizer directement
(compatible transformers 4.x et 5.x).
"""
import numpy as np

DEFAULT_READER = "deepset/roberta-base-squad2"


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def best_span(start_logits, end_logits, offsets, seq_ids, context, max_answer_len=30):
    """Cherche la meilleure reponse dans UN morceau de texte tokenise.
    Retourne (texte, score_span, score_null). Les scores sont des probabilites (start x end)."""
    s, e = np.array(start_logits, dtype=float), np.array(end_logits, dtype=float)
    valid = np.array([sid == 1 for sid in seq_ids])          # tokens du passage (pas la question)
    allowed = valid.copy()
    allowed[0] = True                                         # <s> sert au score "pas de reponse"
    s[~allowed], e[~allowed] = -1e9, -1e9
    ps, pe = _softmax(s), _softmax(e)
    null = float(ps[0] * pe[0])

    best, best_span_ij = 0.0, None
    idx = np.where(valid)[0]
    for i in idx:
        for j in idx[(idx >= i) & (idx < i + max_answer_len)]:
            sc = ps[i] * pe[j]
            if sc > best:
                best, best_span_ij = float(sc), (i, j)
    if best_span_ij is None:
        return "", 0.0, null
    i, j = best_span_ij
    return context[offsets[i][0]:offsets[j][1]].strip(), best, null


class Reader:
    def __init__(self, model=DEFAULT_READER, max_answer_len=30, max_length=384, stride=128):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model).to(self.device).eval()
        self.max_answer_len, self.max_length, self.stride = max_answer_len, max_length, stride

    def _read_one(self, question, context):
        """Lit un passage (decoupe en morceaux chevauchants s'il est trop long)."""
        enc = self.tokenizer(question, context, truncation="only_second", max_length=self.max_length,
                             stride=self.stride, return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding=True)
        best_text, best_score, best_null = "", 0.0, 0.0
        n = len(enc["input_ids"])
        inputs = {"input_ids": self.torch.tensor(enc["input_ids"]).to(self.device),
                  "attention_mask": self.torch.tensor(enc["attention_mask"]).to(self.device)}
        with self.torch.no_grad():
            out = self.model(**inputs)
        for c in range(n):
            text, score, null = best_span(out.start_logits[c].cpu().numpy(), out.end_logits[c].cpu().numpy(),
                                          enc["offset_mapping"][c], enc.sequence_ids(c), context, self.max_answer_len)
            if score > best_score:
                best_text, best_score = text, score
            best_null = max(best_null, null)
        return best_text, best_score, best_null

    def answer(self, question, passages, min_score=0.0, allow_abstain=True):
        """passages : liste de {"id", "text"}. Retourne la reponse la plus sure parmi tous les passages.
        answer == "" signifie 'je ne sais pas' (le modele juge que 'pas de reponse' est plus probable).
        allow_abstain=False : force la meilleure reponse meme si 'pas de reponse' gagne (pour le diagnostic)."""
        best = {"answer": "", "score": 0.0, "passage_id": None}
        for p in passages:
            text, score, null = self._read_one(question, p["text"])
            if text and (score > null or not allow_abstain) and score > best["score"]:   # "pas de reponse" doit etre battu
                best = {"answer": text, "score": float(score), "passage_id": p["id"]}
        if best["score"] < min_score:
            best["answer"] = ""
        return best