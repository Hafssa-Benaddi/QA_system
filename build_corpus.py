"""
ETAPE 3a : corpus de recherche propre.

    python build_corpus.py

Entree : passages.json
Sorties :
    corpus.json  -> passages sans menus de navigation et sans doublons exacts
    id_map.json  -> ancien passage_id -> passage_id conserve (pour retrouver le passage de reference)
"""
import re

from common import load_json, save_json

# Meme detection que audit_dataset.py (menus de navigation / historique de versions de git-scm.com)
NAV_MARKERS = ["Latest version ▾", "Topics ▾", "Plumbing Commands", "Getting and Creating Projects",
               "Basic Snapshotting", "last updated in"]


def is_nav(text):
    hits = sum(m in text for m in NAV_MARKERS)
    return hits >= 2 or bool(re.search(r"\d+\.\d+\.\d+ → \d+\.\d+\.\d+ no changes", text))


def norm_text(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def main():
    passages = load_json("passages.json")
    corpus, id_map, seen, n_nav = [], {}, {}, 0
    for p in passages:
        if is_nav(p["text"]):
            n_nav += 1
            continue
        key = norm_text(p["text"])
        if key in seen:                       # doublon exact : on renvoie vers le premier
            id_map[p["id"]] = seen[key]
            continue
        seen[key] = p["id"]
        id_map[p["id"]] = p["id"]
        corpus.append(p)
    save_json("corpus.json", corpus)
    save_json("id_map.json", id_map)
    print(f"Passages d'origine : {len(passages)}")
    print(f"  menus de navigation retires : {n_nav}")
    print(f"  doublons retires            : {len(passages) - n_nav - len(corpus)}")
    print(f"Corpus final : {len(corpus)} passages -> corpus.json")


if __name__ == "__main__":
    main()