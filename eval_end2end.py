"""
ETAPE 5b - EVALUATION COMPLETE : Exact Match et F1.

    python eval_end2end.py --method hybrid --ks 1,3,5 --split test

Deux configurations :
    oracle     : le Reader recoit le BON passage (borne haute : qualite du Reader seul)
    retrieved  : le Reader recoit les K passages du retriever (le vrai systeme)
L'ecart entre les deux = la perte due au retriever.
Option --limit N pour un test rapide (la lecture est lente sur CPU).
"""
import argparse

from common import exact_match, f1_score, load_json, pick_qa_file, save_json
from reader import DEFAULT_READER, Reader
from retrievers import DEFAULT_MODEL, build_retriever


def score(preds, qa):
    n = len(qa)
    em = sum(exact_match(p["answer"], e["answer"]) for p, e in zip(preds, qa)) / n
    f1 = sum(f1_score(p["answer"], e["answer"]) for p, e in zip(preds, qa)) / n
    abst = sum(p["answer"] == "" for p in preds) / n
    return {"EM": round(em, 3), "F1": round(f1, 3), "sans_reponse": round(abst, 3), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=0.0, help="seuil en dessous duquel le systeme repond 'je ne sais pas'")
    ap.add_argument("--no-abstain", action="store_true", help="diagnostic : force une reponse meme si le modele prefere 'pas de reponse'")
    ap.add_argument("--qa", default=None)
    a = ap.parse_args()

    corpus = load_json("corpus.json")
    by_id = {p["id"]: p for p in corpus}
    qa = load_json(pick_qa_file(a.qa))
    if a.split != "all":
        qa = [e for e in qa if e["split"] == a.split]
    if a.limit:
        qa = qa[:a.limit]
    print(f"{len(qa)} questions ({a.split}) | retriever={a.method} | reader={a.reader}")

    reader, retr = Reader(a.reader), build_retriever(a.method, corpus, a.model)
    results, details = {}, {}

    preds = [reader.answer(e["question"], [{"id": e["passage_id"], "text": e["context"]}], a.min_score, not a.no_abstain) for e in qa]
    results["oracle"] = score(preds, qa)
    details["oracle"] = preds
    print("oracle     ", results["oracle"])

    for k in [int(x) for x in a.ks.split(",")]:
        preds = []
        for e in qa:
            passages = [by_id[retr.ids[i]] for i in retr.search(e["question"], k)]
            preds.append(reader.answer(e["question"], passages, a.min_score, not a.no_abstain))
        key = f"retrieved_top{k}"
        results[key], details[key] = score(preds, qa), preds
        print(f"{key:<11}", results[key])

    save_json("end2end_results.json", {"config": vars(a), "results": results,
              "predictions": [{"id": e["id"], "question": e["question"], "gold": e["answer"],
                               **{k: v[i]["answer"] for k, v in details.items()}} for i, e in enumerate(qa)]})
    print("\nDetails (question, reponse attendue, reponses predites) : end2end_results.json")


if __name__ == "__main__":
    main()