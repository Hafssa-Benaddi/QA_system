"""Fonctions communes : fichiers JSON, normalisation et metriques EM / F1."""
import json
import re
from collections import Counter
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_qa_file(explicit=None):
    """Prend qa_dataset_final.json (apres relecture) s'il existe, sinon qa_dataset_clean.json."""
    if explicit:
        return Path(explicit)
    for name in ("qa_dataset_final.json", "qa_dataset_clean.json"):
        if Path(name).exists():
            return Path(name)
    raise FileNotFoundError("Aucun fichier qa_dataset_final.json / qa_dataset_clean.json")


def normalize(s: str) -> str:
    """Normalisation adaptee a la doc technique : on garde la casse et les tirets
    (-a et -A, --quiet et quiet sont differents), on retire seulement les
    guillemets/ponctuation de bord et les espaces multiples."""
    s = s.strip().strip("`'\".,;:").strip()
    return re.sub(r"\s+", " ", s)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)