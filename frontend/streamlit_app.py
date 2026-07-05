"""
Minimal Streamlit UI per BLUEPRINT.md §4 (chosen over a custom frontend for
demo-ability ROI). Two panels: upload documents, then ask questions and see
answers with citations and a confidence score.
"""
import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Knowledge Base", layout="wide")
st.title("📚 RAG Knowledge Base")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("PDF only", type=["pdf"])
    if uploaded_file is not None and st.button("Ingest document"):
        with st.spinner("Parsing, chunking, embedding, indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                if result["skipped_duplicate"]:
                    st.info(f"'{result['filename']}' was already indexed (unchanged).")
                else:
                    st.success(
                        f"Indexed {result['chunks_indexed']} chunks from "
                        f"{result['page_count']} pages of '{result['filename']}'."
                    )
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")

st.header("Ask a question")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["citations"]:
            with st.expander("Sources"):
                for c in turn["citations"]:
                    st.markdown(f"- **[Source {c['source_index']}]** {c['doc_title']}, p.{c['page_number']}")
        st.caption(f"Confidence: {turn['confidence']:.2f}" +
                   (f" · fallback: {turn['fallback_reason']}" if turn["fallback_reason"] else ""))
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("👍", key=f"up_{turn['log_id']}"):
                requests.post(f"{API_URL}/feedback", json={"log_id": turn["log_id"], "rating": "up"})
                st.toast("Thanks for the feedback!")
        with col2:
            if st.button("👎", key=f"down_{turn['log_id']}"):
                requests.post(f"{API_URL}/feedback", json={"log_id": turn["log_id"], "rating": "down"})
                st.toast("Thanks for the feedback!")

question = st.chat_input("Ask something about your uploaded documents...")
if question:
    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.append({
                "question": question,
                "answer": data["answer"],
                "citations": data["citations"],
                "confidence": data["confidence"],
                "fallback_reason": data["fallback_reason"],
                "log_id": data["log_id"],
            })
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Query failed: {e}")
