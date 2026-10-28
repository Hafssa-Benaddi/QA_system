"""
ETAPE 3 - REPRESENTATION : generation des embeddings des passages.

Install :  pip install sentence-transformers numpy
Lancer  :  python build_corpus.py                       # une fois (corpus.json)
           python embed.py                              # modele par defaut (BGE small)
           python embed.py --model sentence-transformers/all-MiniLM-L6-v2
           python embed.py --check                      # test rapide : 3 requetes -> 3 passages les plus proches

Sorties (dans embeddings/) :
    <modele>.npy       matrice [n_passages, dimension], vecteurs normalises (norme 1)
    <modele>_ids.json  liste des passage_id, dans le meme ordre que les lignes de la matrice

Comme les vecteurs sont normalises, le produit scalaire (emb @ q) EST la similarite cosinus.
"""
import argparse
from pathlib import Path

import numpy as np

from common import load_json, save_json

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
OUT_DIR = Path("embeddings")

# Certains modeles exigent un prefixe different pour les questions et pour les passages.
# (prefixe_question, prefixe_passage)
PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}


def slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_queries(model, model_name, queries):
    prefix = PREFIXES.get(model_name, ("", ""))[0]
    return model.encode([prefix + q for q in queries], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")


def ensure_embeddings(model_name, corpus, batch_size=32, force=False, model=None):
    """Charge les embeddings du disque s'ils existent (et correspondent au corpus), sinon les calcule."""
    OUT_DIR.mkdir(exist_ok=True)
    npy, ids_file = OUT_DIR / f"{slug(model_name)}.npy", OUT_DIR / f"{slug(model_name)}_ids.json"
    ids = [p["id"] for p in corpus]
    if npy.exists() and ids_file.exists() and not force and load_json(ids_file) == ids:
        print(f"Embeddings deja calcules : {npy}")
        return np.load(npy)

    model = model or load_model(model_name)
    model.max_seq_length = min(getattr(model, "max_seq_length", 512), 512)
    prefix = PREFIXES.get(model_name, ("", ""))[1]
    texts = [prefix + p["text"] for p in corpus]
    print(f"Encodage de {len(texts)} passages avec {model_name} ...")
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype("float32")
    np.save(npy, emb)
    save_json(ids_file, ids)
    print(f"Sauvegarde : {npy}  forme={emb.shape}")
    return emb


def sanity_check(model_name, corpus, emb, model):
    """3 requetes d'essai : verifie visuellement que la recherche semantique fonctionne."""
    queries = ["How do I stash only part of my changes?",
               "Which option shows the abbreviated commit hash?",
               "What does git gc do?"]
    q = encode_queries(model, model_name, queries)
    for query, qv in zip(queries, q):
        top = np.argsort(-(emb @ qv))[:3]
        print(f"\nQ: {query}")
        for rank, i in enumerate(top, 1):
            print(f"  {rank}. [{corpus[i]['source']}] {corpus[i]['text'][:110]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="recalcule meme si le fichier existe")
    ap.add_argument("--check", action="store_true", help="test rapide apres calcul")
    a = ap.parse_args()

    corpus = load_json(a.corpus)
    model = load_model(a.model)
    emb = ensure_embeddings(a.model, corpus, a.batch_size, a.force, model)
    print(f"Matrice : {emb.shape[0]} passages x {emb.shape[1]} dimensions | norme moyenne = {np.linalg.norm(emb, axis=1).mean():.3f}")
    if a.check:
        sanity_check(a.model, corpus, emb, model)


if __name__ == "__main__":
    main()