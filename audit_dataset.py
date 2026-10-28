"""
Audit de qa_dataset.json : trouve les problemes que le controle manuel ne voit pas.

    python audit_dataset.py

Sorties :
    qa_dataset_clean.json   -> dataset nettoye (a utiliser pour la suite)
    removed_by_audit.json   -> paires retirees, avec la raison
Les paires "a surveiller" restent dans le dataset mais portent un champ "flags".
"""

import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("qa_dataset.json")
CLEAN = Path("qa_dataset_clean.json")
REMOVED = Path("removed_by_audit.json")

# Texte typique des passages "menu de navigation / historique de versions" de git-scm.com
NAV_MARKERS = ["Latest version ▾", "Topics ▾", "Plumbing Commands", "Getting and Creating Projects",
               "Basic Snapshotting", "last updated in"]
VAGUE_MARKERS = ["this page", "on this page", "the above", "mentioned above", "in this section"]


def norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


def is_nav(context: str) -> bool:
    hits = sum(m in context for m in NAV_MARKERS)
    return hits >= 2 or bool(re.search(r"\d+\.\d+\.\d+ → \d+\.\d+\.\d+ no changes", context))


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    kept, removed = [], []
    seen = set()
    reasons = Counter()

    def drop(ex, reason):
        removed.append({**ex, "removed_reason": reason})
        reasons[reason] += 1

    for ex in data:
        ctx, ans, q = ex["context"], ex["answer"], ex["question"]
        flags = []

        # 1. Doublons exacts (meme passage, meme question)
        key = (ex["passage_id"], norm(q))
        if key in seen:
            drop(ex, "doublon (meme passage et meme question)")
            continue
        seen.add(key)

        # 2. Passage = menu de navigation ou historique de versions
        if is_nav(ctx):
            drop(ex, "passage de navigation / historique de versions")
            continue

        # 3. Question qui depend de la page
        if any(m in q.lower() for m in VAGUE_MARKERS):
            drop(ex, "question dependante de la page")
            continue

        # 4. Reponse absente du passage
        if ans not in ctx:
            drop(ex, "reponse absente du passage")
            continue

        # 5. Position de la reponse fausse (frequent avec les reponses courtes : find() prend la 1re occurrence)
        start = ex.get("answer_start", -1)
        if ctx[start:start + len(ans)] != ans:
            ex = {**ex, "answer_start": ctx.find(ans)}
            flags.append("span corrige")
        n_occ = ctx.count(ans)
        if n_occ > 1 and len(ans) <= 4:
            flags.append(f"span ambigu ({n_occ} occurrences)")

        # 6. Reponses ultra courtes (EM trivial, peu informatif)
        if len(ans.strip()) <= 2:
            flags.append("reponse tres courte")

        if flags:
            ex = {**ex, "flags": flags}
            for f in flags:
                reasons["a surveiller : " + f.split(" (")[0]] += 1
        kept.append(ex)

    CLEAN.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    REMOVED.write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Avant : {len(data)} paires | Apres : {len(kept)} paires | Retirees : {len(removed)}\n")
    for r, n in reasons.most_common():
        print(f"  {n:4d}  {r}")
    print("\nDocuments couverts :", len({e['doc'] for e in kept}))
    print("Test :", sum(e.get("split") == "test" for e in kept), "| Dev :", sum(e.get("split") == "dev" for e in kept))
    if len(kept) < 500:
        print("\nAttention : moins de 500 paires apres nettoyage. Genere de nouveaux lots (batch_31 et plus).")


if __name__ == "__main__":
    main()