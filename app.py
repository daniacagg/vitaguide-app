"""
app.py — Accessible Streamlit Frontend for VitaGuide (Exercise 3).
Deployed on Streamlit Community Cloud (Free Tier) with multi-turn memory,
Qdrant topic filtering, starter prompts, educational disclaimer, and
deterministic WHO/PAHO source attribution.
"""

from __future__ import annotations

import os
import uuid
import streamlit as st
from backend import VitaGuideBackend

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="VitaGuide — Public Health Promotion Chatbot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)


def sync_streamlit_secrets() -> None:
    """Safely exposes Streamlit Cloud secrets as environment variables."""
    try:
        for key in ("OMNIROUTE_API_KEY", "OMNIROUTE_BASE_URL", "QDRANT_URL", "QDRANT_API_KEY"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key]).strip()
    except Exception:
        pass


sync_streamlit_secrets()


@st.cache_resource(show_spinner="Initializing FastEmbed ONNX model and connecting to Qdrant...")
def get_backend() -> VitaGuideBackend:
    return VitaGuideBackend(config_path="config.yaml")


try:
    backend = get_backend()
except Exception as exc:
    st.error(f"Startup Configuration Error: {exc}")
    st.info(
        "Configure `OPENROUTER_API_KEY` (plus optional `QDRANT_URL` and `QDRANT_API_KEY`) "
        "in Streamlit Community Cloud under **Settings → Secrets**."
    )
    st.stop()

cfg = backend.config

# ------------------------------------------------------------------------------
# Session Initialization (Anonymous ID — Zero PII)
# ------------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

TOPIC_OPTIONS = {
    "All WHO Topics": "all",
    "Physical Activity": "physical_activity",
    "Healthy Diet": "healthy_diet",
    "Hypertension Prevention": "hypertension",
    "Dengue Prevention": "dengue_prevention",
    "Mental Well-being & Sleep": "mental_wellbeing",
}

SAMPLE_PROMPTS = [
    "What are the WHO weekly physical activity recommendations for adults?",
    "What are the recommended daily limits for salt and free sugars?",
    "How can households prevent Aedes mosquitoes from breeding?",
    "What daily habits does WHO recommend for managing stress and sleep?",
]

# ------------------------------------------------------------------------------
# Sidebar: Preferences, Topic Filter, User Guidance & Runtime Info
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title(f"🩺 {cfg['app']['name']}")
    st.caption(cfg["app"]["subtitle"])

    st.subheader("Conversation Controls")
    selected_topic_label = st.selectbox(
        "Filter knowledge base by topic (Qdrant metadata):",
        options=list(TOPIC_OPTIONS.keys()),
        index=0,
    )
    selected_topic = TOPIC_OPTIONS[selected_topic_label]

    detail_level = st.radio(
        "Response style preference:",
        options=["concise", "detailed"],
        format_func=lambda x: "Concise summary" if x == "concise" else "Detailed explanation",
        index=0,
    )
    backend.store.set_profile(
        raw_session_id=st.session_state.session_id,
        detail_level=detail_level,
        preferred_topic=None if selected_topic == "all" else selected_topic,
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        backend.memory.clear(st.session_state.session_id)
        st.rerun()

    st.divider()
    with st.expander("ℹ️ Capabilities & Limitations", expanded=True):
        st.markdown(
            """
            **What VitaGuide can do:**
            - Share evidence-based preventive guidance from public WHO/PAHO fact sheets.
            - Handle multi-turn follow-up questions within your session.
            - Filter answers by public health topic and cite retrieved WHO sources.

            **What VitaGuide cannot do:**
            - Diagnose medical conditions or interpret personal lab tests.
            - Prescribe or recommend medication dosages.
            - Replace emergency services or licensed healthcare professionals.
            """
        )

    with st.expander("⚙️ System Configuration (Read-Only)", expanded=False):
        stats = backend.store.get_summary_stats()
        st.markdown(
            f"""
            - **Open LLM:** `{backend.llm.model_id}`
            - **Embeddings:** `{cfg['embeddings']['model_name']}` (`fastembed` ONNX)
            - **Vector DB:** `{backend.qdrant_mode}`
            - **Top-k / Threshold:** `{cfg['qdrant']['top_k']}` / `{cfg['qdrant']['score_threshold']}`
            - **Temperature:** `{cfg['llm']['temperature']}`
            - **Logged Turns:** `{stats['total_interactions']}`
            """
        )

# ------------------------------------------------------------------------------
# Main Header & Visible Educational Disclaimer
# ------------------------------------------------------------------------------
st.title(f"🩺 {cfg['app']['name']} — Public Health Education Assistant")
st.warning(cfg["app"]["disclaimer"])

if not st.session_state.messages:
    st.markdown("**Try one of these recommended prompts to get started:**")
    cols = st.columns(2)
    for idx, prompt_text in enumerate(SAMPLE_PROMPTS):
        if cols[idx % 2].button(prompt_text, key=f"starter_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

# ------------------------------------------------------------------------------
# Render Multi-Turn Chat History
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            if meta.get("sources"):
                with st.expander("📚 Verified WHO/PAHO Sources Used"):
                    for src in meta["sources"]:
                        st.markdown(f"- {src}")
            st.caption(
                f"Status: `{meta['status']}` | "
                f"Similarity: `{meta['top_score']}` | "
                f"Latency: `{meta['latency_ms'] / 1000:.2f}s` | "
                f"Engine: `{meta['model_used']}`"
            )

# ------------------------------------------------------------------------------
# Handle New User Turn (Typed Input or Starter Prompt Button)
# ------------------------------------------------------------------------------
user_query = st.chat_input(
    "Ask a preventive health question (e.g., WHO diet, exercise, dengue, or stress habits)..."
)
if st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Searching WHO guidelines and generating grounded response..."):
            try:
                result = backend.chat(
                    session_id=st.session_state.session_id,
                    user_input=user_query,
                    topic_filter=selected_topic,
                )
            except Exception as exc:
                st.error(f"Temporary inference error: {exc}")
                st.stop()

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📚 Verified WHO/PAHO Sources Used"):
                for src in result["sources"]:
                    st.markdown(f"- {src}")
        st.caption(
            f"Status: `{result['status']}` | "
            f"Similarity: `{result['top_score']}` | "
            f"Latency: `{result['latency_ms'] / 1000:.2f}s` | "
            f"Engine: `{result['model_used']}`"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "status": result["status"],
                "sources": result["sources"],
                "top_score": result["top_score"],
                "latency_ms": result["latency_ms"],
                "model_used": result["model_used"],
            },
        }
    )
