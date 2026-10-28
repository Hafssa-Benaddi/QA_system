"""
ETAPE D sans cle API : generation des questions avec un chat gratuit (copier-coller).
Ecrit dans candidates.json, donc les etapes E (review) et F (export) de phase1_dataset.py
fonctionnent sans changement.

A placer a cote de phase1_dataset.py.

    python generate_free.py export     # cree batches/batch_01.txt, batch_02.txt ...
    -> colle chaque fichier dans un chat gratuit (Claude, ChatGPT...), sauvegarde la reponse JSON
       dans answers/batch_01.json, answers/batch_02.json ...
    python generate_free.py import     # verifie et ajoute tout dans candidates.json
"""

import json
import random
import re
import sys
from pathlib import Path

from phase1_dataset import CANDIDATES_FILE, PASSAGES_FILE, PROMPT

BATCH_DIR = Path("batches")
ANSWERS_DIR = Path("answers")


# ----------------------------------------------------------------------
# Fonctions communes
# ----------------------------------------------------------------------

def load_passages(shuffle=True):
    passages = json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
    if shuffle:
        random.seed(42)  # meme ordre a chaque execution, donc les lots restent coherents
        random.shuffle(passages)
    return passages


def load_candidates():
    if CANDIDATES_FILE.exists():
        return json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    return []


def save_candidates(candidates):
    CANDIDATES_FILE.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_list(raw: str):
    """Extrait la premiere liste JSON trouvee dans une reponse de modele."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", raw, flags=re.S)
    return json.loads(match.group(0)) if match else []


def add_pairs(candidates, passage, pairs):
    """Garde une paire seulement si la reponse apparait exactement dans le passage."""
    added = 0
    for pair in pairs:
        q = str(pair.get("question", "")).strip()
        ans = str(pair.get("answer", "")).strip()
        start = passage["text"].find(ans) if ans else -1
        if q and start != -1:
            candidates.append({
                "passage_id": passage["id"], "source": passage["source"], "passage": passage["text"],
                "question": q, "answer": ans, "answer_start": start,
            })
            added += 1
    return added


# ----------------------------------------------------------------------
# Copier-coller avec un chat gratuit
# ----------------------------------------------------------------------

BATCH_HEADER = """You are building an extractive question answering dataset from technical documentation.

Below are {n_passages} passages, each with an ID. For EACH passage, write {per_passage} different questions
that can be answered using ONLY that passage.

Rules:
- Each question must be understandable on its own (never say "in this passage").
- The answer must be an EXACT, short substring copied character-for-character from the passage
  (a command, a value, a name, a short phrase; max 15 words).
- Vary the question types (how, what, which, when, why).

Return ONLY one JSON list, with no other text, in this format:
[{{"passage_id": "...", "question": "...", "answer": "..."}}]

"""


def export_batches(passages_per_batch: int = 8, per_passage: int = 3, max_passages: int = 400):
    BATCH_DIR.mkdir(exist_ok=True)
    ANSWERS_DIR.mkdir(exist_ok=True)
    done = {c["passage_id"] for c in load_candidates()}
    passages = [p for p in load_passages()[:max_passages] if p["id"] not in done]

    if not passages:
        print("Rien a exporter : tous les passages (dans la limite max_passages) ont deja des candidats.")
        print("Pour en generer plus, augmente max_passages dans export_batches().")
        return

    # Numerotation : on continue apres les lots deja existants pour ne rien ecraser
    existing = [int(m.group(1)) for f in BATCH_DIR.glob("batch_*.txt")
                if (m := re.search(r"batch_(\d+)", f.name))]
    start_num = max(existing, default=0) + 1

    n_files = 0
    for b, i in enumerate(range(0, len(passages), passages_per_batch), start=start_num):
        chunk = passages[i:i + passages_per_batch]
        text = BATCH_HEADER.format(n_passages=len(chunk), per_passage=per_passage)
        for p in chunk:
            text += f'PASSAGE ID: {p["id"]}\n"""{p["text"]}"""\n\n'
        (BATCH_DIR / f"batch_{b:02d}.txt").write_text(text, encoding="utf-8")
        n_files += 1

    print(f"{n_files} fichiers de lots crees dans {BATCH_DIR}/")
    print(f"Pour chacun : colle-le dans un chat, puis sauvegarde la reponse JSON dans {ANSWERS_DIR}/ avec le meme numero")
    print("(batch_01.txt -> answers/batch_01.json).")


def import_answers():
    passages = {p["id"]: p for p in load_passages(shuffle=False)}
    candidates = load_candidates()
    seen = {(c["passage_id"], c["question"]) for c in candidates}
    total_added = bad_id = duplicates = not_found = 0

    for path in sorted(ANSWERS_DIR.glob("*.json")):
        try:
            pairs = parse_json_list(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{path.name}: lecture impossible ({e})")
            continue
        for pair in pairs:
            p = passages.get(pair.get("passage_id"))
            question = str(pair.get("question", "")).strip()
            if not p:
                bad_id += 1          # passage_id inconnu ou absent
            elif (p["id"], question) in seen:
                duplicates += 1      # deja importe
            elif add_pairs(candidates, p, [pair]):
                total_added += 1
                seen.add((p["id"], question))
            else:
                not_found += 1       # reponse non copiee exactement depuis le passage

    save_candidates(candidates)
    print(f"ajoutes : {total_added}")
    print(f"doublons (deja importes) : {duplicates}")
    print(f"reponse absente du passage : {not_found}")
    print(f"passage_id inconnu : {bad_id}")
    print(f"{len(candidates)} candidats dans {CANDIDATES_FILE}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    commands = {"export": export_batches, "import": import_answers}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print("Usage: python generate_free.py [export | import]")
        sys.exit(1)
    commands[sys.argv[1]]()