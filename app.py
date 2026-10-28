"""
ETAPE 6 - APPLICATION : interface QA interactive sur la documentation Git.

Install :  pip install streamlit
Lancer  :  streamlit run app.py
"""
import streamlit as st

from qa_system import QASystem, doc_url, highlight

st.set_page_config(page_title="Git QA", page_icon="🔎", layout="centered")
st.title("🔎 Git QA")
st.caption("Pose une question sur Git : le systeme cherche dans la documentation officielle et en extrait la reponse.")


@st.cache_resource(show_spinner="Chargement des modeles (premiere fois seulement)...")
def load_system(method: str) -> QASystem:
    return QASystem(method=method)


with st.sidebar:
    st.header("Reglages")
    method = st.selectbox("Retriever", ["bm25", "hybrid", "dense"], help="bm25 = meilleur sur nos tests")
    k = st.slider("Passages lus par le Reader (K)", 1, 10, 5)
    system = load_system(method)
    threshold = st.slider("Seuil de confiance", 0.0, 1.0, float(system.threshold), 0.01,
                          help="En dessous de ce score, le systeme repond 'je ne sais pas'.")
    show_retrieved = st.checkbox("Afficher tous les passages trouves", value=False)

EXAMPLES = ["How do I stash only part of my changes?", "What does git gc do?",
            "Which option shows the abbreviated commit hash?", "What is the capital of France?"]
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Ta question", key="question", placeholder="e.g. How do I undo the last commit?")

if question.strip():
    with st.spinner("Recherche et lecture..."):
        r = system.ask(question.strip(), k=k, min_score=threshold)

    if r["answered"]:
        st.success(f"**{r['answer']}**")
        st.progress(min(r["score"], 1.0), text=f"Confiance du modele : {r['score']:.2f} (seuil {r['threshold']:.2f})")
        p = r["passage"]
        st.markdown(f"**Source :** [{p['source']}]({doc_url(p['source'])})")
        st.markdown(f"<div style='font-size:0.9rem;line-height:1.5'>{highlight(p['text'], r['answer'])}</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("Je ne trouve pas la reponse dans la documentation Git.")
        st.caption(f"Confiance trop faible : {r['score']:.3f} < seuil {r['threshold']:.2f}")
        if r["best_guess"]:
            with st.expander("Meilleure hypothese du modele (peu fiable)"):
                st.write(r["best_guess"])

    if show_retrieved:
        st.subheader("Passages trouves")
        for i, p in enumerate(r["retrieved"], 1):
            with st.expander(f"{i}. {p['source']}"):
                st.write(p["text"])