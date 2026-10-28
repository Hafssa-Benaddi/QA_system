"""
Choix du seuil de confiance : en dessous de ce score, le systeme repond "je ne sais pas".

    python tune_threshold.py --method bm25 --k 5 --limit 200

Important : le dataset ne contient QUE des questions avec reponse. Si on ne regardait que lui,
le meilleur seuil serait toujours 0 (s'abstenir ne peut que faire perdre des points). On ajoute donc
des questions HORS SUJET (sans reponse dans la doc Git) : un bon systeme doit s'abstenir dessus.

On regle le seuil sur les questions DEV (jamais sur le test), puis on le fixe pour le test final.
Sortie : threshold.json  +  tableau seuil -> couverture / precision / reponses aux questions hors sujet.
"""
import argparse

from common import exact_match, f1_score, load_json, pick_qa_file, save_json
from reader import DEFAULT_READER, Reader
from retrievers import DEFAULT_MODEL, build_retriever

OFF_TOPIC = [
    "What is the capital of France?", "Who wrote Romeo and Juliet?", "How many legs does a spider have?",
    "What is the boiling point of water?", "Which planet is closest to the sun?", "How do I bake a chocolate cake?",
    "Who painted the Mona Lisa?", "What is the largest ocean on Earth?", "When did World War II end?",
    "What is the square root of 144?", "How do I change a flat tire?", "What language is spoken in Brazil?",
    "How many players are on a football team?", "What is the tallest mountain in the world?",
    "Which vitamin does sunlight give us?", "What is the speed of light?", "Who discovered penicillin?",
    "What is the currency of Japan?", "How do I brew green tea?", "What is the chemical symbol for gold?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bm25", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-offtopic-answered", type=float, default=0.10,
                    help="part maximale acceptee de questions hors sujet qui recoivent quand meme une reponse")
    ap.add_argument("--qa", default=None)
    a = ap.parse_args()

    corpus = load_json("corpus.json")
    by_id = {p["id"]: p for p in corpus}
    dev = [e for e in load_json(pick_qa_file(a.qa)) if e["split"] == "dev"][:a.limit]
    reader, retr = Reader(a.reader), build_retriever(a.method, corpus, a.model)

    def run(question):
        passages = [by_id[retr.ids[i]] for i in retr.search(question, a.k)]
        return reader.answer(question, passages, 0.0, allow_abstain=False)   # meilleure reponse + son score

    print(f"{len(dev)} questions dev + {len(OFF_TOPIC)} questions hors sujet ...")
    on = [(run(e["question"]), e) for e in dev]
    off = [run(q) for q in OFF_TOPIC]

    thresholds = [0, .01, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9]
    rows, chosen = [], None
    print(f"\n{'seuil':>6} {'couverture':>11} {'EM repondues':>13} {'EM globale':>11} {'hors sujet repondues':>21}")
    for t in thresholds:
        answered = [(p, e) for p, e in on if p["score"] >= t and p["answer"]]
        cov = len(answered) / len(on)
        em_ans = sum(exact_match(p["answer"], e["answer"]) for p, e in answered) / max(len(answered), 1)
        f1_ans = sum(f1_score(p["answer"], e["answer"]) for p, e in answered) / max(len(answered), 1)
        em_all = sum(exact_match(p["answer"], e["answer"]) for p, e in answered) / len(on)   # abstention = point perdu
        off_rate = sum(p["score"] >= t and bool(p["answer"]) for p in off) / len(off)
        rows.append({"seuil": t, "couverture": cov, "EM_repondues": em_ans, "F1_repondues": f1_ans,
                     "EM_globale": em_all, "hors_sujet_repondues": off_rate})
        if chosen is None and off_rate <= a.max_offtopic_answered:
            chosen = t
        print(f"{t:6.2f} {cov:11.1%} {em_ans:13.3f} {em_all:11.3f} {off_rate:21.1%}")

    chosen = chosen if chosen is not None else thresholds[-1]
    save_json("threshold.json", {"min_score": chosen, "method": a.method, "k": a.k,
                                 "regle": f"plus petit seuil avec <= {a.max_offtopic_answered:.0%} de questions hors sujet repondues",
                                 "table": rows})
    print(f"\nSeuil retenu : {chosen}  (ecrit dans threshold.json)")
    print(f"Test final :  python eval_end2end.py --method {a.method} --ks {a.k} --split test --no-abstain --min-score {chosen}")


if __name__ == "__main__":
    main()