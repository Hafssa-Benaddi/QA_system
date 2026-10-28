"""
ETAPE 5a - EVALUATION DU RETRIEVER : Recall@K et MRR.

    python eval_retrieval.py --methods bm25,dense,hybrid
    python eval_retrieval.py --methods bm25 --qa qa_dataset_clean.json

Deux facons de compter un succes :
    strict : le passage de reference (passage_id) est dans les K premiers
    souple : au moins un des K passages contient la reponse attendue
             (utile car la meme info est repetee dans plusieurs pages Git)
"""
import argparse
from pathlib import Path

from common import load_json, pick_qa_file, save_json
from retrievers import DEFAULT_MODEL, build_retriever


def first_rank(hits):
    return next((r for r, h in enumerate(hits) if h), None)


def metrics(ranks, ks):
    n = len(ranks)
    out = {f"R@{k}": sum(r is not None and r < k for r in ranks) / n for k in ks}
    out["MRR"] = sum(1 / (r + 1) for r in ranks if r is not None) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="bm25,dense,hybrid")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--qa", default=None)
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--ks", default="1,3,5,10")
    a = ap.parse_args()

    ks = [int(k) for k in a.ks.split(",")]
    corpus, id_map = load_json(a.corpus), load_json("id_map.json")
    qa = load_json(pick_qa_file(a.qa))
    print(f"{len(qa)} questions | {len(corpus)} passages dans le corpus\n")
    subsets = {"toutes": lambda e: True, "dev": lambda e: e["split"] == "dev", "test": lambda e: e["split"] == "test",
               "non ambigues": lambda e: not e.get("ambiguous")}
    Path("runs").mkdir(exist_ok=True)
    results = {}

    for method in a.methods.split(","):
        retr = build_retriever(method, corpus, a.model)
        strict, soft, run = [], [], {}
        for e in qa:
            gold = id_map.get(e["passage_id"], e["passage_id"])
            top = retr.search(e["question"], max(ks))
            strict.append(first_rank([retr.ids[i] == gold for i in top]))
            soft.append(first_rank([e["answer"] in retr.texts[i] for i in top]))
            run[e["id"]] = [retr.ids[i] for i in top]
        save_json(f"runs/{method}.json", run)

        results[method] = {}
        for sname, cond in subsets.items():
            idx = [i for i, e in enumerate(qa) if cond(e)]
            if not idx:
                continue
            results[method][sname] = {"strict": metrics([strict[i] for i in idx], ks),
                                      "souple": metrics([soft[i] for i in idx], ks), "n": len(idx)}
        print(f"=== {method} ===")
        print(f"{'sous-ensemble':<14}{'mode':<8}" + "".join(f"{k:>8}" for k in list(results[method]['toutes']['strict'])) + "      n")
        for sname, r in results[method].items():
            for mode in ("strict", "souple"):
                print(f"{sname:<14}{mode:<8}" + "".join(f"{v:8.3f}" for v in r[mode].values()) + f"  {r['n']:5d}")
        print()
    save_json("retrieval_results.json", results)
    print("Resultats detailles : retrieval_results.json | classements : runs/<methode>.json")


if __name__ == "__main__":
    main()