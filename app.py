import logging
import streamlit as st

import config
from ingest import index_file, SUPPORTED_EXTENSIONS
from rag_engine import stream_ask, ask_cached
from vector_store import get_collection_stats, list_documents, delete_document, delete_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

st.set_page_config(page_title="PDF Q&A with Local RAG", layout="centered", initial_sidebar_state="expanded", menu_items=None)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: #0f1117;
}

section[data-testid="stSidebar"] {
    background: #161922;
    border-right: 1px solid #1e2230;
}

section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #6c5ce7, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 500;
    transition: all 0.2s;
}

section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.4);
}

section[data-testid="stSidebar"] .stButton button[pkind="secondary"] {
    background: transparent;
    border: 1px solid #2a2e3a;
    color: #8b8fa3;
}

section[data-testid="stSidebar"] .stButton button[pkind="secondary"]:hover {
    border-color: #6c5ce7;
    color: #c4c7d5;
}

div.stChatMessage {
    background: #1a1d27;
    border: 1px solid #222636;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}

div.stChatMessage[data-testid="user-message"] {
    background: linear-gradient(135deg, #1e2233, #1a1d27);
    border-color: #2a2e42;
}

[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: white !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: #2a2e3a !important;
    color: #a0a3b5 !important;
}

.stChatInputContainer {
    background: #1a1d27 !important;
    border: 1px solid #222636 !important;
    border-radius: 14px !important;
    padding: 0.3rem !important;
}

.stChatInputContainer input {
    background: transparent !important;
    color: #e0e2eb !important;
    font-family: 'Inter', sans-serif !important;
}

.stChatInputContainer input::placeholder {
    color: #5a5e72 !important;
}

.stChatInputContainer:focus-within {
    border-color: #6c5ce7 !important;
    box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.15) !important;
}

h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #e0e2eb, #a0a3b5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem !important;
}

.stExpander {
    border: none !important;
    background: #1a1d27 !important;
    border-radius: 12px !important;
    margin: 0.4rem 0;
}

.stExpander summary {
    color: #8b8fa3 !important;
    font-size: 0.85rem !important;
    font-weight: 500;
}

.stExpander div[data-testid="stExpanderContent"] {
    border-top: 1px solid #222636 !important;
    padding: 0.8rem !important;
}

[data-testid="stStatusWidget"] {
    background: #1a1d27 !important;
    border: 1px solid #222636 !important;
    border-radius: 10px !important;
}

.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

div[data-testid="stBottom"] {
    background: linear-gradient(180deg, transparent, #0f1117 50%);
    padding-top: 1.5rem;
}

[data-testid="stToolbarActions"] { display: none !important; }
header button[kind="header"] { display: none !important; }

.doc-item {
    background: #1a1d27;
    border: 1px solid #222636;
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.doc-item-name {
    color: #c4c7d5;
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    margin-right: 0.5rem;
}

.doc-item-chunks {
    color: #5a5e72;
    font-size: 0.7rem;
    white-space: nowrap;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "ollama_ok" not in st.session_state:
    st.session_state.ollama_ok = None

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("PDF Q&A with Local RAG")
    st.markdown('<p style="color:#5a5e72; font-size:0.9rem; margin-top:-0.5rem;">Chat with your PDFs — powered by local AI</p>', unsafe_allow_html=True)
with col2:
    backend = "GEMINI" if use_gemini else "LOCAL"
    st.markdown(f'<div style="margin-top:1.2rem; text-align:right;"><span style="background:#1a1d27; color:#34d399; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.75rem; font-weight:600; border:1px solid #1e2a24;">● {backend}</span></div>', unsafe_allow_html=True)


use_gemini = bool(config.GEMINI_API_KEY)

if use_gemini:
    st.session_state.ollama_ok = True
elif st.session_state.ollama_ok is None:
    with st.spinner("Checking Ollama connection..."):
        try:
            import requests
            r = requests.get(config.OLLAMA_BASE_URL + "/api/tags", timeout=5)
            st.session_state.ollama_ok = r.ok
        except Exception:
            st.session_state.ollama_ok = False

if not st.session_state.ollama_ok and not use_gemini:
    st.warning("Ollama is not reachable. Make sure `ollama serve` is running.", icon="⚠️")


with st.sidebar:
    st.markdown('<h3 style="color:#e0e2eb; font-weight:600; margin-bottom:0.2rem;">Documents</h3>', unsafe_allow_html=True)

    doc_count = get_collection_stats()
    st.markdown(
        f"""
        <div style="background:#1a1d27; border:1px solid #222636; border-radius:12px; padding:0.8rem 1rem; margin:0.8rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#8b8fa3; font-size:0.85rem;">Total chunks</span>
                <span style="color:#e0e2eb; font-weight:700; font-size:1.3rem;">{doc_count}</span>
            </div>
            <div style="margin-top:0.5rem; height:3px; background:#222636; border-radius:4px; overflow:hidden;">
                <div style="height:100%; width:{min(doc_count / max(doc_count, 1) * 100, 100)}%; background:linear-gradient(90deg, #6c5ce7, #a855f7); border-radius:4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    doc_list = list_documents()
    if doc_list:
        st.markdown('<p style="color:#5a5e72; font-size:0.8rem; margin-bottom:0.3rem;">Indexed files</p>', unsafe_allow_html=True)
        for d in doc_list:
            col_a, col_b, col_c = st.columns([0.6, 0.2, 0.2])
            with col_a:
                st.markdown(f'<span style="color:#c4c7d5; font-size:0.8rem;">{d["name"]}</span>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<span style="color:#5a5e72; font-size:0.75rem;">{d["chunks"]}ch</span>', unsafe_allow_html=True)
            with col_c:
                if st.button("✕", key=f"del_{d['name']}", help=f"Remove {d['name']}"):
                    delete_document(d["name"])
                    st.rerun()

    st.markdown('<p style="color:#5a5e72; font-size:0.8rem; margin:0.5rem 0 0.3rem 0;">Upload file</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a file",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Index file", use_container_width=True):
            st.session_state.processing = True
            with st.spinner("Indexing..."):
                try:
                    count = index_file(uploaded)
                except ValueError as e:
                    st.error(str(e))
                    st.session_state.processing = False
                    st.stop()
            st.session_state.processing = False
            if count == 0:
                st.warning("No text extracted from this file.")
            else:
                st.success(f"Indexed {count} chunks.")
                st.rerun()

    if doc_count > 0:
        st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)
        if st.button("Clear all", use_container_width=True, type="secondary"):
            delete_all()
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#5a5e72; line-height:1.6;">
            <span style="color:#6c5ce7;">Ollama</span> · ChromaDB<br>
            llama3.2:3b · nomic-embed-text<br>
            cache enabled · streaming
        </div>
        """,
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    score_pct = min(int(s["score"] * 100), 99)
                    st.markdown(
                        f"""
                        <div style="background:#0f1117; border:1px solid #222636; border-radius:10px; padding:0.6rem 0.8rem; margin:0.4rem 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                                <span style="color:#a0a3b5; font-size:0.8rem; font-weight:500;">{s['source']}</span>
                                <span style="color:#6c5ce7; font-size:0.75rem; font-weight:600;">{score_pct}%</span>
                            </div>
                            <p style="color:#5a5e72; font-size:0.8rem; margin:0; line-height:1.5;">{s["content"][:300]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if prompt := st.chat_input("Ask a question about your documents...", disabled=st.session_state.processing):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_response = ""
        sources = []

        for token, final_sources in stream_ask(prompt):
            if token:
                full_response += token
                stream_placeholder.markdown(full_response + "▌")
            if final_sources is not None:
                sources = final_sources
                stream_placeholder.markdown(full_response)

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for s in sources:
                    score_pct = min(int(s["score"] * 100), 99)
                    st.markdown(
                        f"""
                        <div style="background:#0f1117; border:1px solid #222636; border-radius:10px; padding:0.6rem 0.8rem; margin:0.4rem 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                                <span style="color:#a0a3b5; font-size:0.8rem; font-weight:500;">{s['source']}</span>
                                <span style="color:#6c5ce7; font-size:0.75rem; font-weight:600;">{score_pct}%</span>
                            </div>
                            <p style="color:#5a5e72; font-size:0.8rem; margin:0; line-height:1.5;">{s["content"][:300]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })
