"""
backend.py — Lightweight, Production-Oriented RAG, Memory, Guardrails, Profiling,
and Logging Backend for VitaGuide (Streamlit Community Cloud Free Tier Ready).
Uses exclusively free, open-source models and libraries (no paid APIs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

load_dotenv()

# ------------------------------------------------------------------------------
# Public, Non-Sensitive WHO / PAHO Knowledge Base (auto-ingested if KB is empty)
# ------------------------------------------------------------------------------
WHO_PUBLIC_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "topic": "physical_activity",
        "source": "WHO Guidelines on Physical Activity and Sedentary Behaviour",
        "year": 2020,
        "url": "https://www.who.int/publications/i/item/9789240015128",
        "content": (
            "Adults aged 18 to 64 years should do at least 150 to 300 minutes of "
            "moderate-intensity aerobic physical activity per week, or at least 75 to "
            "150 minutes of vigorous-intensity aerobic physical activity, or an "
            "equivalent combination of both throughout the week. For additional health "
            "benefits, adults should also perform muscle-strengthening activities at "
            "moderate or greater intensity that involve all major muscle groups on 2 "
            "or more days a week. Adults should limit the amount of time spent being "
            "sedentary. Replacing sedentary time with physical activity of any "
            "intensity, including light intensity such as standing or casual walking, "
            "provides measurable health benefits. Children and adolescents aged 5 to "
            "17 years should do at least an average of 60 minutes per day of "
            "moderate-to-vigorous intensity, mostly aerobic, physical activity across "
            "the week. Older adults aged 65 and above should add varied multicomponent "
            "physical activity that emphasizes functional balance and strength training "
            "on 3 or more days a week to enhance functional capacity and prevent falls."
        ),
    },
    {
        "topic": "healthy_diet",
        "source": "WHO Healthy Diet Fact Sheet",
        "year": 2020,
        "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "content": (
            "A healthy diet helps protect against malnutrition in all its forms, as "
            "well as noncommunicable diseases including diabetes, heart disease, "
            "stroke, and cancer. For adults, a healthy diet includes eating at least "
            "400 grams (five portions) of fruit and vegetables per day, excluding "
            "potatoes, sweet potatoes, cassava, and other starchy roots. Free sugars "
            "should make up less than 10 percent of total energy intake, which is "
            "equivalent to about 50 grams (around 12 level teaspoons) for a person of "
            "healthy body weight consuming about 2000 calories per day; reducing free "
            "sugars to below 5 percent of total energy intake provides additional "
            "health benefits. Total fat intake should not exceed 30 percent of total "
            "energy intake. Saturated fats should be less than 10 percent of total "
            "energy intake, and trans-fats should be less than 1 percent, with a shift "
            "toward unsaturated fats found in fish, avocado, nuts, and olive or "
            "sunflower oils. Salt intake should be kept to less than 5 grams per day "
            "(equivalent to less than 2000 mg of sodium per day), which helps prevent "
            "hypertension and reduces the risk of heart disease and stroke. Practical "
            "ways to reduce sodium include cooking with herbs and spices instead of "
            "salt, removing salt shakers from the table, limiting salty sauces such as "
            "soy sauce, and choosing fresh foods over ultra-processed snacks."
        ),
    },
    {
        "topic": "hypertension",
        "source": "WHO Hypertension Fact Sheet",
        "year": 2023,
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "content": (
            "Hypertension, or high blood pressure, occurs when the pressure in blood "
            "vessels is too high (140/90 mmHg or higher). Many people with "
            "hypertension do not notice symptoms, which is why regular blood pressure "
            "checks by a qualified healthcare worker are essential. Modifiable risk "
            "factors include unhealthy diets high in salt and saturated fats, physical "
            "inactivity, consumption of tobacco and alcohol, and being overweight or "
            "obese. Lifestyle changes that help prevent and lower high blood pressure "
            "include eating a balanced diet rich in fruits and vegetables, reducing "
            "salt intake to under 5 grams per day, being physically active every week, "
            "maintaining a healthy body weight, avoiding tobacco use, limiting alcohol "
            "consumption, and managing chronic stress. Only a licensed healthcare "
            "provider can diagnose hypertension or prescribe blood pressure medication."
        ),
    },
    {
        "topic": "dengue_prevention",
        "source": "WHO/PAHO Dengue and Severe Dengue Fact Sheet",
        "year": 2024,
        "url": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
        "content": (
            "Dengue is a viral infection transmitted to humans through the bite of "
            "infected Aedes mosquitoes, primarily Aedes aegypti, which typically bite "
            "during the daytime, especially early morning and late afternoon. Because "
            "these mosquitoes breed in clean standing water around homes, community "
            "prevention focuses on eliminating breeding sites: empty, scrub, and cover "
            "domestic water storage containers weekly; dispose of solid waste such as "
            "unused tires, bottles, and buckets that collect rainwater; and clean roof "
            "gutters and flower pot trays regularly. Personal protective measures "
            "include wearing long-sleeved shirts and long pants, using EPA-registered "
            "insect repellents containing DEET, picaridin, or IR3535, installing "
            "window and door screens, and using mosquito nets when sleeping during the "
            "day. Warning signs of severe dengue—such as severe abdominal pain, "
            "persistent vomiting, rapid breathing, bleeding gums or nose, or extreme "
            "fatigue—require urgent medical attention at a health facility."
        ),
    },
    {
        "topic": "mental_wellbeing",
        "source": "WHO Doing What Matters in Times of Stress & Mental Health Promotion",
        "year": 2022,
        "url": "https://www.who.int/publications/i/item/9789240003927",
        "content": (
            "WHO highlights practical everyday habits that support mental well-being "
            "and stress management for the general population. Maintaining a regular "
            "daily routine—including consistent sleep and wake times, balanced meals, "
            "and scheduled physical movement—helps regulate mood and energy. Grounding "
            "techniques, such as slow deep breathing and noticing physical sensations "
            "in the present moment, can help interrupt cycles of overwhelming worry. "
            "Limiting continuous exposure to distressing news or social media, staying "
            "socially connected with trusted friends or family, and dedicating time to "
            "enjoyable or meaningful activities strengthen psychological resilience. "
            "For sleep hygiene, adults generally benefit from 7 to 9 hours of sleep "
            "per night, avoiding caffeine late in the day, and reducing bright screen "
            "exposure before bedtime. Anyone experiencing persistent distress, "
            "hopelessness, or thoughts of self-harm should seek immediate support from "
            "a licensed mental health professional or local crisis helpline."
        ),
    },
]

# ------------------------------------------------------------------------------
# Preprocessing, PII Scrubbing & Deterministic Clinical Guardrails
# ------------------------------------------------------------------------------
PII_PATTERNS = [
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{8,12}\b"), "[REDACTED_ID]"),
]

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "severe bleeding", "unconscious", "stroke symptoms", "suicidal",
    "kill myself", "self-harm", "overdose", "heart attack",
]

CLINICAL_OUT_OF_SCOPE_PATTERNS = [
    r"\bdiagnos(e|is)\b",
    r"\bprescrib(e|ed|ing)\b",
    r"\b(dosage|dose|mg)\b.*\b(pill|tablet|medication|medicine|drug|losartan|metformin|ibuprofen|aspirin)\b",
    r"\bwhat medication should i take\b",
    r"\bshould i stop taking my\b",
    r"\binterpret my (lab|blood test|x-ray|mri)\b",
]

VAGUE_EXACT_QUERIES = {
    "help", "hi", "hello", "health", "how do i stay safe", "how do i stay safe?",
    "what should i eat", "what should i eat?", "advice", "tips", "stay healthy",
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_and_scrub(user_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", user_text or "").strip()[:1000]
    for pattern, replacement in PII_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def check_safety_and_ambiguity(
    cleaned_query: str, has_history: bool
) -> tuple[str, Optional[str]]:
    q_lower = cleaned_query.lower()

    if any(kw in q_lower for kw in EMERGENCY_KEYWORDS):
        return (
            "emergency_escalation",
            (
                "**Urgent Safety Notice:** It sounds like you may be experiencing a "
                "medical or mental health emergency. VitaGuide is an educational assistant "
                "and cannot provide emergency medical care. **Please call your local "
                "emergency number immediately (e.g., 123 in Colombia, 911 in the US) or "
                "go to the nearest emergency health facility.**"
            ),
        )

    for pat in CLINICAL_OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q_lower):
            return (
                "clinical_refusal",
                (
                    "I cannot diagnose medical conditions, interpret personal lab tests, "
                    "or recommend specific medications or dosages. VitaGuide only provides "
                    "general preventive health information based on public WHO/PAHO "
                    "guidelines. **Please consult a licensed physician or pharmacist for "
                    "personalized medical advice.**"
                ),
            )

    word_count = len(cleaned_query.split())
    if (q_lower in VAGUE_EXACT_QUERIES) or (word_count <= 2 and not has_history):
        return (
            "ambiguity_clarification",
            (
                "Could you clarify what preventive health topic you would like to explore? "
                "I can share public WHO/PAHO guidelines on:\n"
                "- **Physical activity** (weekly exercise recommendations)\n"
                "- **Healthy diet** (daily fruit/vegetable, sugar, fat, and sodium limits)\n"
                "- **Hypertension prevention** (lifestyle habits to protect blood pressure)\n"
                "- **Dengue prevention** (eliminating mosquito breeding sites & warning signs)\n"
                "- **Mental well-being & sleep hygiene** (stress management routines)"
            ),
        )

    return "ok", None


# ------------------------------------------------------------------------------
# Refined System Prompt (incorporating Exercise 2 error analysis findings)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are VitaGuide, a helpful, empathetic public health education assistant.

STRICT RULES:
1. Answer ONLY using facts explicitly stated inside the <retrieved_context> tags below.
2. NEVER add extra food items, ingredients, statistics, or recommendations not present in <retrieved_context>.
3. Preserve exact units and terminology from <retrieved_context> (e.g., distinguish grams of salt from milligrams of sodium).
4. NEVER diagnose illnesses, prescribe medications, or replace a licensed healthcare professional.
5. If <retrieved_context> does not contain enough evidence, say so explicitly and refer to https://www.who.int.
6. Response style preference: {style_instruction}
7. End your response with a line starting with "Source:" listing the exact guideline name(s) used.
"""


def build_rag_messages(
    user_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]],
    detail_level: str = "concise",
) -> List[Dict[str, str]]:
    style_instruction = (
        "Provide a brief, clear summary (3–5 bullet points or one short paragraph)."
        if detail_level == "concise"
        else "Provide a clear, structured explanation covering all relevant practical points from the context."
    )

    context_blocks = [
        f"[Source {i}: {item['source']} ({item['year']}) | Topic: {item['topic']}]\n{item['text']}"
        for i, item in enumerate(retrieved_chunks, start=1)
    ]
    formatted_context = "\n\n".join(context_blocks)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(style_instruction=style_instruction),
        }
    ]
    messages.extend(history_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                f"<retrieved_context>\n{formatted_context}\n</retrieved_context>\n\n"
                f"<user_question>\n{user_query}\n</user_question>\n\n"
                "Answer <user_question> grounded strictly in <retrieved_context>."
            ),
        }
    )
    return messages


# ------------------------------------------------------------------------------
# Free Open-Source LLM Client (OpenRouter :free Tier via Standard HTTP Requests)
# ------------------------------------------------------------------------------
class OpenSourceFreeLLMClient:
    """
    Connects to a self-hosted, MIT-licensed OmniRoute AI gateway using its
    OpenAI-compatible /v1/chat/completions endpoint and free open-source routes.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        self.api_key = os.getenv("OMNIROUTE_API_KEY", "").strip()
        raw_url = os.getenv("OMNIROUTE_BASE_URL", llm_config["base_url"]).strip().rstrip("/")
        if raw_url.endswith("/v1/chat/completions"):
            self.base_url = raw_url
        elif raw_url.endswith("/v1"):
            self.base_url = f"{raw_url}/chat/completions"
        else:
            self.base_url = f"{raw_url}/v1/chat/completions"

        self.models = [llm_config["primary_model"]] + list(
            llm_config.get("fallback_models", [])
        )
        self.model_id = self.models[0]
        self.temperature = float(llm_config["temperature"])
        self.max_new_tokens = int(llm_config["max_new_tokens"])
        self.timeout = int(llm_config.get("timeout_seconds", 45))

    @staticmethod
    def _clean_response(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        return cleaned.strip()

    def generate(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for candidate_model in self.models:
            payload = {
                "model": candidate_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:250]}"
                    time.sleep(0.8)
                    continue

                data = resp.json()
                raw_content = data["choices"][0]["message"].get("content", "")
                content = self._clean_response(raw_content)
                if content:
                    actual_model = data.get("model", candidate_model)
                    self.model_id = actual_model
                    return content, actual_model
                last_error = f"Empty response returned by {candidate_model}"
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.8)

        raise RuntimeError(
            f"All OmniRoute free routes failed ({self.base_url}). Last error: {last_error}"
        )


# ------------------------------------------------------------------------------
# Short-Term Rolling Memory & Anonymized SQLite/JSONL Store
# ------------------------------------------------------------------------------
class ShortTermMemoryBuffer:
    def __init__(self, max_turns: int = 4, max_chars: int = 2400):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._sessions: Dict[str, deque] = {}

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns)
        self._sessions[session_id].append({"user": user_msg, "assistant": assistant_msg})

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        turns = list(self._sessions.get(session_id, []))
        messages: List[Dict[str, str]] = []
        total_chars = 0
        for turn in reversed(turns):
            turn_len = len(turn["user"]) + len(turn["assistant"])
            if total_chars + turn_len > self.max_chars:
                break
            messages.insert(0, {"role": "assistant", "content": turn["assistant"]})
            messages.insert(0, {"role": "user", "content": turn["user"]})
            total_chars += turn_len
        return messages

    def get_last_user_query(self, session_id: str) -> str:
        turns = self._sessions.get(session_id)
        return turns[-1]["user"] if turns else ""

    def has_history(self, session_id: str) -> bool:
        return bool(self._sessions.get(session_id))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class StateAndAnalyticsStore:
    def __init__(self, db_path: str, jsonl_path: str):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def anonymize_session_id(raw_session_id: str) -> str:
        return hashlib.sha256(raw_session_id.encode("utf-8")).hexdigest()[:12]

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_profiles (
                    session_hash TEXT PRIMARY KEY,
                    detail_level TEXT NOT NULL DEFAULT 'concise',
                    preferred_topic TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    session_hash TEXT NOT NULL,
                    topic_filter TEXT,
                    status TEXT NOT NULL,
                    top_similarity REAL,
                    chunks_retrieved INTEGER,
                    model_used TEXT,
                    latency_ms REAL,
                    query_length INTEGER
                )
                """
            )

    def set_profile(
        self,
        raw_session_id: str,
        detail_level: str = "concise",
        preferred_topic: Optional[str] = None,
    ) -> None:
        s_hash = self.anonymize_session_id(raw_session_id)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO session_profiles (session_hash, detail_level, preferred_topic, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_hash) DO UPDATE SET
                    detail_level=excluded.detail_level,
                    preferred_topic=excluded.preferred_topic,
                    updated_at=excluded.updated_at
                """,
                (s_hash, detail_level, preferred_topic, now),
            )

    def get_profile(self, raw_session_id: str) -> Dict[str, Any]:
        s_hash = self.anonymize_session_id(raw_session_id)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT detail_level, preferred_topic FROM session_profiles WHERE session_hash = ?",
                (s_hash,),
            ).fetchone()
        if row:
            return {"detail_level": row[0], "preferred_topic": row[1]}
        return {"detail_level": "concise", "preferred_topic": None}

    def log_interaction(
        self,
        raw_session_id: str,
        topic_filter: Optional[str],
        status: str,
        top_similarity: float,
        chunks_retrieved: int,
        model_used: str,
        latency_ms: float,
        query_length: int,
        sources: List[str],
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_hash": self.anonymize_session_id(raw_session_id),
            "topic_filter": topic_filter,
            "status": status,
            "top_similarity": round(float(top_similarity), 4),
            "chunks_retrieved": chunks_retrieved,
            "model_used": model_used,
            "latency_ms": round(float(latency_ms), 1),
            "query_length": query_length,
            "sources": sources,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interaction_logs (
                    timestamp_utc, session_hash, topic_filter, status,
                    top_similarity, chunks_retrieved, model_used,
                    latency_ms, query_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp_utc"],
                    record["session_hash"],
                    record["topic_filter"],
                    record["status"],
                    record["top_similarity"],
                    record["chunks_retrieved"],
                    record["model_used"],
                    record["latency_ms"],
                    record["query_length"],
                ),
            )
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_summary_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM interaction_logs").fetchone()[0]
            avg_rag_ms = conn.execute(
                "SELECT AVG(latency_ms) FROM interaction_logs WHERE status = 'grounded_rag'"
            ).fetchone()[0]
        return {
            "total_interactions": int(total),
            "avg_rag_latency_s": round(float(avg_rag_ms) / 1000.0, 2) if avg_rag_ms else 0.0,
        }


# ------------------------------------------------------------------------------
# End-to-End Orchestrator
# ------------------------------------------------------------------------------
class VitaGuideBackend:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.collection_name = self.config["qdrant"]["collection_name"]
        self.embedder = TextEmbedding(model_name=self.config["embeddings"]["model_name"])
        self.qdrant, self.qdrant_mode = self._init_qdrant()
        self._ensure_knowledge_base_indexed()
        self.llm = OpenSourceFreeLLMClient(self.config["llm"])
        self.memory = ShortTermMemoryBuffer(
            max_turns=self.config["memory"]["max_turns"],
            max_chars=self.config["memory"]["max_history_chars"],
        )
        self.store = StateAndAnalyticsStore(
            db_path=self.config["storage"]["sqlite_db_path"],
            jsonl_path=self.config["storage"]["jsonl_log_path"],
        )

    def _init_qdrant(self) -> tuple[QdrantClient, str]:
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if qdrant_url and qdrant_api_key:
            return (
                QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30),
                "Qdrant Cloud (Free Tier)",
            )
        fallback_path = self.config["qdrant"]["local_fallback_path"]
        return QdrantClient(path=fallback_path), f"Local Qdrant ({fallback_path})"

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.embedder.embed(texts)]

    def _ensure_knowledge_base_indexed(self) -> None:
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config["embeddings"]["vector_size"],
                    distance=models.Distance.COSINE,
                ),
            )
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="topic",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        count = self.qdrant.count(collection_name=self.collection_name, exact=True).count
        if count > 0:
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        records = []
        for doc in WHO_PUBLIC_DOCUMENTS:
            for idx, text in enumerate(splitter.split_text(doc["content"])):
                records.append(
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['source']}_{idx}")),
                        "text": text.strip(),
                        "topic": doc["topic"],
                        "source": doc["source"],
                        "year": doc["year"],
                        "url": doc["url"],
                        "chunk_index": idx,
                    }
                )
        vectors = self._encode_texts([r["text"] for r in records])
        points = [
            models.PointStruct(id=r["id"], vector=v, payload={k: r[k] for k in r if k != "id"})
            for r, v in zip(records, vectors)
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _build_retrieval_query(self, session_id: str, current_query: str) -> str:
        last_q = self.memory.get_last_user_query(session_id)
        follow_up_markers = ("it", "that", "those", "first", "second", "more", "what about")
        if last_q and (
            len(current_query.split()) <= 10
            or any(m in current_query.lower() for m in follow_up_markers)
        ):
            return f"{last_q} {current_query}"
        return current_query

    def retrieve_context(
        self, query_text: str, topic_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self._encode_texts([query_text])[0]
        q_filter = None
        if topic_filter and topic_filter != "all":
            q_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic", match=models.MatchValue(value=topic_filter)
                    )
                ]
            )

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=self.config["qdrant"]["top_k"],
            score_threshold=self.config["qdrant"]["score_threshold"],
            with_payload=True,
        )

        return [
            {
                "score": float(pt.score),
                "text": (pt.payload or {}).get("text", ""),
                "source": (pt.payload or {}).get("source", "WHO Public Guideline"),
                "topic": (pt.payload or {}).get("topic", "general"),
                "year": (pt.payload or {}).get("year", ""),
                "url": (pt.payload or {}).get("url", ""),
            }
            for pt in response.points
        ]

    def chat(
        self,
        session_id: str,
        user_input: str,
        topic_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        cleaned_query = preprocess_and_scrub(user_input)

        profile = self.store.get_profile(session_id)
        active_topic = topic_filter if topic_filter != "all" else None
        detail_level = profile.get("detail_level", "concise")

        status, canned_reply = check_safety_and_ambiguity(
            cleaned_query, has_history=self.memory.has_history(session_id)
        )
        if status != "ok" and canned_reply:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.store.log_interaction(
                session_id, active_topic, status, 0.0, 0, "rule_guardrail",
                latency_ms, len(cleaned_query), [],
            )
            return {
                "answer": canned_reply,
                "status": status,
                "sources": [],
                "retrieved_chunks": [],
                "top_score": 0.0,
                "model_used": "rule_guardrail",
                "latency_ms": round(latency_ms, 1),
            }

        retrieval_query = self._build_retrieval_query(session_id, cleaned_query)
        retrieved_chunks = self.retrieve_context(retrieval_query, topic_filter=active_topic)

        if not retrieved_chunks:
            fallback_msg = (
                "I do not have verified WHO/PAHO guideline passages in my current "
                "knowledge base to answer that question reliably. To avoid providing "
                "unverified health claims, please consult the official WHO portal "
                "(https://www.who.int) or speak with a licensed healthcare professional."
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.store.log_interaction(
                session_id, active_topic, "fallback_no_context", 0.0, 0,
                "rag_threshold_fallback", latency_ms, len(cleaned_query), [],
            )
            return {
                "answer": fallback_msg,
                "status": "fallback_no_context",
                "sources": [],
                "retrieved_chunks": [],
                "top_score": 0.0,
                "model_used": "rag_threshold_fallback",
                "latency_ms": round(latency_ms, 1),
            }

        history_msgs = self.memory.get_messages(session_id)
        messages = build_rag_messages(
            cleaned_query, retrieved_chunks, history_msgs, detail_level=detail_level
        )
        answer_text, model_used = self.llm.generate(messages)
        self.memory.add_turn(session_id, cleaned_query, answer_text)

        unique_sources = list(dict.fromkeys(c["source"] for c in retrieved_chunks))
        top_score = retrieved_chunks[0]["score"]
        latency_ms = (time.perf_counter() - start_time) * 1000

        self.store.log_interaction(
            session_id, active_topic, "grounded_rag", top_score,
            len(retrieved_chunks), model_used, latency_ms,
            len(cleaned_query), unique_sources,
        )

        return {
            "answer": answer_text,
            "status": "grounded_rag",
            "sources": unique_sources,
            "retrieved_chunks": retrieved_chunks,
            "top_score": round(top_score, 4),
            "model_used": model_used,
            "latency_ms": round(latency_ms, 1),
        }
