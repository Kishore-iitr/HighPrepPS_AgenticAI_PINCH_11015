"""
Agentic Event Planner — LangGraph-based multi-agent orchestration
Built with: Streamlit, LangGraph, LangChain, ChromaDB, OpenRouter, Tavily, LangSmith
"""

import base64
import builtins as _builtins
import html
import json
import os
import re
import sqlite3
import smtplib
import shutil
import sys
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

from pricing_engine import PricingEngine


load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agentic Event Planner",
    page_icon="AE",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Agentic Event Planner"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "default"))
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

VECTOR_DB_PATHS = {
    "sponsor": "./vector_db",
    "speaker": "./speaker_vector_db",
    "exhibitor": "./exhibitor_vector_db",
    "venue": "./venue_vector_db",
    "community": "./community_vector_db",
    "event_ops": "./event_ops_vector_db",
    "pricing": "./pricing_vector_db",
    "contact": "./contact_vector_db",
}

CONTACTS_CSV_PATH = "data/contacts.csv"

DATASET_SPECS = {
    "events": {"label": "Events", "path": "data/events_v2.csv"},
    "sponsors": {"label": "Sponsors", "path": "data/sponsors_v2.csv"},
    "speakers": {"label": "Speakers", "path": "data/speakers_v2.csv"},
    "exhibitors": {"label": "Exhibitors", "path": "data/exhibitors.csv"},
    "venues": {"label": "Venues", "path": "data/venues.csv"},
    "communities": {"label": "Communities", "path": "data/communities.csv"},
    "sessions": {"label": "Sessions", "path": "data/sessions.csv"},
    "rooms": {"label": "Rooms", "path": "data/rooms.csv"},
    "time_slots": {"label": "Time Slots", "path": "data/time_slots.csv"},
    "pricing_tiers": {"label": "Pricing Tiers", "path": "data/pricing_tiers.csv"},
    "contacts": {"label": "Contacts", "path": "data/contacts.csv"},
}

INGESTION_STATUS_PATH = "data/.ingestion_status.json"
SESSION_HISTORY_DIR = os.getenv("SESSION_HISTORY_DIR", "data/session_history")

CUSTOM_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(255, 140, 66, 0.18), transparent 30%),
      radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 28%),
      linear-gradient(180deg, #fff9f2 0%, #fffdf8 38%, #f7f3ed 100%);
  }
  .hero {
    padding: 1.35rem 1.4rem;
    border: 1px solid rgba(68, 47, 26, 0.12);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.74);
    box-shadow: 0 18px 40px rgba(79, 55, 30, 0.08);
  }
  .hero h1 {
    font-size: 2.1rem;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
  }
  .hero p {
    margin: 0;
    color: rgba(50, 39, 31, 0.78);
    font-size: 1rem;
  }
  .pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    margin: 0 0.25rem 0.25rem 0;
    border-radius: 999px;
    background: rgba(17, 94, 89, 0.09);
    color: #0f5e59;
    font-size: 0.82rem;
    border: 1px solid rgba(17, 94, 89, 0.12);
  }
  .section-card {
    padding: 1rem 1rem 0.7rem 1rem;
    border-radius: 18px;
    border: 1px solid rgba(68, 47, 26, 0.10);
    background: rgba(255, 255, 255, 0.82);
  }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.08rem;
        font-weight: 700;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp div[data-testid="stMarkdownContainer"] p,
    .stApp div[data-testid="stMetricLabel"] {
        font-size: 1.03rem;
    }
    .stApp h1 { font-size: 2.35rem; }
    .stApp h2 { font-size: 1.8rem; }
    .stApp h3 { font-size: 1.35rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict, total=False):
    """State machine for multi-agent event planning orchestration."""
    # Input
    user_input: Dict[str, Any]
    query: str
    
    # Routing
    selected_agent: str
    route_target: str
    
    # Retrieval
    retrieval_query: str
    raw_docs: List[str]
    relevant_docs: List[str]
    context: str
    web_profiles: List[Dict[str, Any]]
    
    # Generation
    sponsors_answer: str
    answer: str
    final_answer: str
    
    # Pricing
    pricing: Dict[str, Any]
    
    # Email/Outreach
    contacts: List[Dict[str, Any]]
    emails: List[Dict[str, Any]]
    email_logs: List[Dict[str, Any]]
    
    # Quality control
    hallucination_verdict: str
    usefulness_verdict: str
    revise_count: int
    rewrite_count: int
    
    # Multi-agent coordination
    required_agents: List[str]
    agent_sequence: List[str]
    orchestration_plan: Dict[str, Any]
    shared_context: str
    agent_outputs: Dict[str, Any]
    logs: List[str]
    execution_logs: List[Dict[str, Any]]
    state_flow: List[str]
    quality_results: List[Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════
# CACHE & HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm() -> Optional[ChatOpenAI]:
    openrouter_api_key = get_runtime_setting("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not openrouter_api_key:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=2500,
    )


@st.cache_resource
def get_vectordb(path: str) -> Optional[Chroma]:
    if not os.path.exists(path):
        return None
    try:
        return Chroma(persist_directory=path, embedding_function=get_embeddings())
    except Exception:
        return None


@st.cache_resource
def get_pricing_engine() -> PricingEngine:
    engine = PricingEngine(
        events_path="data/events_v2.csv",
        venues_path="data/venues.csv",
        pricing_tiers_path="data/pricing_tiers.csv",
        pricing_vector_db_path=VECTOR_DB_PATHS["pricing"],
        embedding_model=EMBED_MODEL,
    )
    engine.load_data()
    engine.preprocess()
    engine.train_model()
    return engine


@st.cache_data
def get_sidebar_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "query": "",
        "category": "",
        "location": "",
        "city": "",
        "audience_size": 0,
        "budget": "",
        "event_topic": "",
        "event_name": "",
    }
    events_path = "data/events_v2.csv"
    if not os.path.exists(events_path):
        return defaults
    try:
        events_df = pd.read_csv(events_path)
    except Exception:
        return defaults
    if events_df.empty:
        return defaults
    first_row = events_df.iloc[0]
    defaults["category"] = safe_text(first_row.get("category"))
    defaults["location"] = safe_text(first_row.get("country"))
    defaults["city"] = safe_text(first_row.get("city"))
    defaults["audience_size"] = int(pd.to_numeric(first_row.get("attendance"), errors="coerce") or 0)
    defaults["event_topic"] = safe_text(first_row.get("category"))
    defaults["event_name"] = safe_text(first_row.get("event_name"))
    query_parts = [part for part in [defaults["category"], defaults["city"], defaults["location"], str(defaults["audience_size"]) if defaults["audience_size"] else ""] if part]
    defaults["query"] = " ".join(query_parts)
    return defaults


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


RUNTIME_SETTING_KEYS = {
    "OPENROUTER_API_KEY": ("OPENROUTER_API_KEY",),
    "TAVILY_API_KEY": ("TAVILY_API_KEY",),
    "LANGSMITH_API_KEY": ("LANGSMITH_API_KEY",),
    "LANGCHAIN_API_KEY": ("LANGCHAIN_API_KEY",),
    "LANGCHAIN_PROJECT": ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT"),
    "LANGCHAIN_ENDPOINT": ("LANGCHAIN_ENDPOINT",),
}


def get_runtime_setting(name: str, default: str = "") -> str:
    session_key = f"runtime_{name.lower()}"
    if session_key in st.session_state:
        return safe_text(st.session_state.get(session_key), default)

    env_keys = RUNTIME_SETTING_KEYS.get(name, (name,))
    for env_key in env_keys:
        value = os.getenv(env_key, "")
        if value:
            return safe_text(value)
    return default


def set_runtime_setting(name: str, value: str) -> None:
    st.session_state[f"runtime_{name.lower()}"] = safe_text(value)


def render_settings_panel() -> None:
    if "settings_panel_open" not in st.session_state:
        st.session_state["settings_panel_open"] = False

    button_label = "Hide Settings" if st.session_state["settings_panel_open"] else "Settings"
    if st.button(button_label, key="toggle_settings_panel", use_container_width=True):
        st.session_state["settings_panel_open"] = not st.session_state["settings_panel_open"]

    if not st.session_state["settings_panel_open"]:
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Runtime Settings")
    st.write("Set the API keys and project name for this session without hardcoding them into the app.")

    col1, col2 = st.columns(2)
    with col1:
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=get_runtime_setting("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
            type="password",
            key="settings_openrouter_api_key",
            placeholder="Enter OpenRouter API key",
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            value=get_runtime_setting("TAVILY_API_KEY", TAVILY_API_KEY),
            type="password",
            key="settings_tavily_api_key",
            placeholder="Enter Tavily API key",
        )
        langsmith_key = st.text_input(
            "LangSmith API Key",
            value=get_runtime_setting("LANGSMITH_API_KEY", LANGSMITH_API_KEY),
            type="password",
            key="settings_langsmith_api_key",
            placeholder="Enter LangSmith API key",
        )
    with col2:
        langchain_key = st.text_input(
            "LangChain API Key",
            value=get_runtime_setting("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY),
            type="password",
            key="settings_langchain_api_key",
            placeholder="Enter LangChain API key",
        )
        langchain_project = st.text_input(
            "LangSmith Project Name",
            value=get_runtime_setting("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT),
            key="settings_langchain_project",
            placeholder="Enter project name",
        )
        langchain_endpoint = st.text_input(
            "LangSmith Endpoint",
            value=get_runtime_setting("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT),
            key="settings_langchain_endpoint",
            placeholder="https://api.smith.langchain.com",
        )

    save_cols = st.columns(3)
    if save_cols[0].button("Save Settings", key="save_runtime_settings", use_container_width=True):
        set_runtime_setting("OPENROUTER_API_KEY", openrouter_key)
        set_runtime_setting("TAVILY_API_KEY", tavily_key)
        set_runtime_setting("LANGSMITH_API_KEY", langsmith_key)
        set_runtime_setting("LANGCHAIN_API_KEY", langchain_key)
        set_runtime_setting("LANGCHAIN_PROJECT", langchain_project)
        set_runtime_setting("LANGCHAIN_ENDPOINT", langchain_endpoint)
        st.session_state["settings_openrouter_api_key"] = openrouter_key
        st.session_state["settings_tavily_api_key"] = tavily_key
        st.session_state["settings_langsmith_api_key"] = langsmith_key
        st.session_state["settings_langchain_api_key"] = langchain_key
        st.session_state["settings_langchain_project"] = langchain_project
        st.session_state["settings_langchain_endpoint"] = langchain_endpoint
        try:
            get_llm.clear()
        except Exception:
            pass
        st.success("Settings saved for this session.")

    if save_cols[1].button("Reset to Environment", key="reset_runtime_settings", use_container_width=True):
        for setting_name in RUNTIME_SETTING_KEYS:
            st.session_state.pop(f"runtime_{setting_name.lower()}", None)
        st.session_state["settings_openrouter_api_key"] = get_runtime_setting("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
        st.session_state["settings_tavily_api_key"] = get_runtime_setting("TAVILY_API_KEY", TAVILY_API_KEY)
        st.session_state["settings_langsmith_api_key"] = get_runtime_setting("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
        st.session_state["settings_langchain_api_key"] = get_runtime_setting("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
        st.session_state["settings_langchain_project"] = get_runtime_setting("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)
        st.session_state["settings_langchain_endpoint"] = get_runtime_setting("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
        try:
            get_llm.clear()
        except Exception:
            pass
        st.success("Session overrides cleared. Environment values will be used.")
        st.rerun()

    if save_cols[2].button("Clear All", key="clear_runtime_settings", use_container_width=True):
        for setting_name in RUNTIME_SETTING_KEYS:
            st.session_state[f"runtime_{setting_name.lower()}"] = ""
        st.session_state["settings_openrouter_api_key"] = ""
        st.session_state["settings_tavily_api_key"] = ""
        st.session_state["settings_langsmith_api_key"] = ""
        st.session_state["settings_langchain_api_key"] = ""
        st.session_state["settings_langchain_project"] = ""
        st.session_state["settings_langchain_endpoint"] = ""
        try:
            get_llm.clear()
        except Exception:
            pass
        st.success("Session values cleared.")

    status_cols = st.columns(3)
    status_cols[0].metric("OpenRouter", "Ready" if safe_text(openrouter_key) else "Missing")
    status_cols[1].metric("Tavily", "Ready" if safe_text(tavily_key) else "Missing")
    status_cols[2].metric("LangSmith", "Ready" if safe_text(langsmith_key or langchain_key) else "Missing")
    st.markdown("</div>", unsafe_allow_html=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


TERMINAL_BUFFER_MAX_LINES = int(os.getenv("TERMINAL_BUFFER_MAX_LINES", "4000"))


def _append_terminal_output_line(line: str) -> None:
    entry = safe_text(line)
    if not entry:
        return
    try:
        if "terminal_output_lines" not in st.session_state:
            st.session_state["terminal_output_lines"] = []
        lines = st.session_state["terminal_output_lines"]
        lines.append(entry)
        if len(lines) > TERMINAL_BUFFER_MAX_LINES:
            del lines[: len(lines) - TERMINAL_BUFFER_MAX_LINES]
    except Exception:
        # Session state might be unavailable in non-Streamlit contexts.
        return


def print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(arg) for arg in args)
    rendered = f"{message}{'' if end is None else str(end)}"

    _builtins.print(*args, **kwargs)

    out_file = kwargs.get("file")
    if out_file is not None and out_file not in (sys.stdout, sys.stderr):
        return
    for line in rendered.replace("\r\n", "\n").split("\n"):
        _append_terminal_output_line(line)


def call_llm(prompt: str, temperature: float = 0.1, retries: int = 2) -> str:
    llm = get_llm()
    if llm is None:
        return "[LLM unavailable] Add an OpenRouter API key in Settings to enable live generation."
    for attempt in range(retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return safe_text(getattr(response, "content", ""))
        except Exception as exc:
            message = str(exc)
            if "429" in message and attempt < retries - 1:
                continue
            if attempt == retries - 1:
                return f"[LLM ERROR] {message}"
    return "[LLM ERROR] Max retries exceeded"


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    tavily_api_key = get_runtime_setting("TAVILY_API_KEY", TAVILY_API_KEY)
    if not tavily_api_key:
        return [{"title": "search_unavailable", "url": "", "snippet": "Add a Tavily API key in Settings to enable web enrichment."}]
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
        )
        hits = []
        for item in response.get("results", []):
            hits.append({
                "title": safe_text(item.get("title")),
                "url": safe_text(item.get("url")),
                "snippet": safe_text(item.get("content")),
            })
        return hits
    except Exception as exc:
        return [{"title": "search_error", "url": "", "snippet": str(exc)}]


def docs_to_text(docs: List[Document], limit: int = 8) -> str:
    return "\n\n".join(doc.page_content for doc in docs[:limit])


def doc_name(doc: Document, fallback: str = "Unknown") -> str:
    md = dict(doc.metadata or {})
    for key in ("name", "company_name", "venue_name", "speaker_name", "community_name", "session_id"):
        if md.get(key):
            return safe_text(md.get(key), fallback)
    text = safe_text(doc.page_content)
    for marker in (" is a ", " exhibited at ", " is located in "):
        if marker in text:
            return text.split(marker)[0].strip()
    return fallback


def build_agent_query(agent: str, user_input: Dict[str, Any]) -> str:
    category = safe_text(user_input.get("category") or user_input.get("event_category") or "conference")
    location = safe_text(user_input.get("location") or user_input.get("city") or "global")
    audience = safe_text(user_input.get("audience_size") or user_input.get("expected_footfall") or "2000")
    topic = safe_text(user_input.get("event_topic") or category)
    budget = safe_text(user_input.get("budget") or "medium")
    builders = {
        "sponsor": f"{category} conference sponsors in {location} audience {audience} budget {budget} marketing spend",
        "speaker": f"{topic} speaker artist candidates in {location} audience {audience} keynote influence",
        "exhibitor": f"companies that exhibited at {category} conferences in {location} startup enterprise tools",
        "venue": f"venues in {location} for {category} events capacity around {audience} budget {budget}",
        "community": f"{category} communities for event promotion in {location} city discord slack linkedin engagement",
        "event_ops": f"event agenda scheduling for {category} in {location} audience {audience} room conflicts speaker slots",
        "pricing": f"ticket pricing tiers and conversion for {category} events in {location} audience {audience} budget {budget}",
    }
    return builders.get(agent, builders["sponsor"])


def retrieve_from_db(agent: str, query: str, k: int = 12) -> List[Document]:
    db = get_vectordb(VECTOR_DB_PATHS[agent])
    if db is None:
        return []
    try:
        return db.similarity_search(query, k=k)
    except Exception:
        return []


@st.cache_resource
def get_contact_vectordb() -> Optional[Chroma]:
    db = get_vectordb(VECTOR_DB_PATHS["contact"])
    if db is not None:
        try:
            if db._collection.count() > 0:
                return db
        except Exception:
            return db
    if not os.path.exists(CONTACTS_CSV_PATH):
        return None
    try:
        contacts_df = pd.read_csv(CONTACTS_CSV_PATH)
        docs: List[Document] = []
        for _, row in contacts_df.iterrows():
            text = (
                f"{safe_text(row.get('name'), 'Unknown Contact')} works at {safe_text(row.get('company'), 'Unknown Company')} "
                f"as {safe_text(row.get('role'), 'Unknown Role')}. "
                f"Type: {safe_text(row.get('type'), 'unknown')}. "
                f"Industry: {safe_text(row.get('industry'), 'unknown')}. "
                f"Email: {safe_text(row.get('email'), 'missing')}. "
                f"LinkedIn: {safe_text(row.get('linkedin'), 'missing')}. "
                f"Relevance score: {safe_text(row.get('relevance_score'), '0')}."
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "contact",
                        "name": safe_text(row.get("name")),
                        "company": safe_text(row.get("company")),
                        "role": safe_text(row.get("role")),
                        "contact_type": safe_text(row.get("type")),
                        "industry": safe_text(row.get("industry")),
                        "email": safe_text(row.get("email")),
                        "linkedin": safe_text(row.get("linkedin")),
                        "relevance_score": safe_text(row.get("relevance_score")),
                    },
                )
            )
        if not docs:
            return None
        built_db = Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            persist_directory=VECTOR_DB_PATHS["contact"],
        )
        built_db.persist()
        return built_db
    except Exception:
        return None


def build_web_profiles(agent: str, docs: List[Document], user_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = []
    seen = set()
    for doc in docs:
        name = doc_name(doc)
        if name not in seen:
            seen.add(name)
            names.append(name)
    profiles: List[Dict[str, Any]] = []
    for name in names[:8]:
        if agent == "sponsor":
            query = f"{name} sponsorship {safe_text(user_input.get('category'), 'conference')} {safe_text(user_input.get('location'), 'global')} recent news"
        elif agent == "speaker":
            query = f"{name} speaker profile keynote publications {safe_text(user_input.get('event_topic'), safe_text(user_input.get('category'), 'AI'))}"
        elif agent == "exhibitor":
            query = f"{name} exhibited at conference expo {safe_text(user_input.get('category'), 'technology')} {safe_text(user_input.get('location'), 'global')}"
        elif agent == "venue":
            query = f"{name} venue rental pricing capacity {safe_text(user_input.get('location'), 'global')} event reviews"
        elif agent == "community":
            query = f"{name} community platform {safe_text(user_input.get('category'), 'technology')} event promotion"
        else:
            query = f"{name} event operations agenda speaker room capacity"
        hits = tavily_search(query, max_results=5)
        joined = " ".join(hit.get("snippet", "") for hit in hits).lower()
        profiles.append({
            "name": name,
            "kind": agent,
            "query": query,
            "hits": hits,
            "signal_score": joined.count("conference") + joined.count("event") + joined.count("speaker") + joined.count("venue") + joined.count("capacity"),
        })
    return profiles


def generate_agent_response(agent: str, user_input: Dict[str, Any], context: str, web_profiles: List[Dict[str, Any]]) -> str:
    labels = {
        "sponsor": "sponsorship strategist",
        "speaker": "programming strategist",
        "exhibitor": "exhibition planning strategist",
        "venue": "venue selection strategist",
        "community": "GTM strategist",
        "event_ops": "event operations planner",
    }
    prompt = (
        f"You are an {labels.get(agent, 'event planner')}.\n\n"
        f"Event details: {json.dumps(user_input, ensure_ascii=True)}\n\n"
        f"Internal context:\n{context}\n\n"
        f"External web evidence:\n{json.dumps(web_profiles, ensure_ascii=True, indent=2)}\n\n"
    )
    if agent == "sponsor":
        prompt += (
            "Task: Recommend sponsors and output strict markdown tables.\n"
            "Include a Sponsor Prioritization Table and a Custom Sponsorship Proposal Table."
        )
    elif agent == "speaker":
        prompt += (
            "Task: Recommend speakers or artists and output strict markdown tables.\n"
            "Include a Speaker/Artist Prioritization Table and an Agenda Mapping Table."
        )
    elif agent == "exhibitor":
        prompt += (
            "Task: Recommend exhibitors and output strict markdown tables.\n"
            "Include an Exhibitor Recommendation Table and an Exhibitor Cluster Table."
        )
    elif agent == "venue":
        prompt += (
            "Task: Recommend venues and output strict markdown tables.\n"
            "Include a Venue Recommendation Table and a Venue Shortlist Rationale Table."
        )
    elif agent == "community":
        prompt += (
            "Task: Recommend communities and GTM distribution steps.\n"
            "Include Community Prioritization, Messaging Strategy, and Distribution Plan tables."
        )
    elif agent == "event_ops":
        prompt += (
            "Task: Build a practical agenda, detect conflicts, and propose resource planning.\n"
            "Use concise execution-ready markdown."
        )
    else:
        prompt += "Task: Produce a concise actionable plan."
    return call_llm(prompt)


def relevance_filter(user_input: Dict[str, Any], docs: List[Document], label: str) -> List[Document]:
    if not docs:
        return []
    if not get_llm():
        return docs[: min(6, len(docs))]
    selected: List[Document] = []
    for doc in docs:
        prompt = (
            f"You are checking whether a {label} document is relevant.\n\n"
            f"Event details: {json.dumps(user_input, ensure_ascii=True)}\n\n"
            f"Document: {doc.page_content}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            selected.append(doc)
    return selected or docs[: min(6, len(docs))]


def choose_agent(user_input: Dict[str, Any]) -> str:
    query = normalize_whitespace(user_input.get("query", "")).lower()
    if any(word in query for word in ["email", "outreach", "contact", "follow up", "send"]):
        return "outreach"
    if any(word in query for word in ["venue", "city", "capacity", "location", "hall"]):
        return "venue"
    if any(word in query for word in ["price", "pricing", "ticket", "revenue", "conversion"]):
        return "pricing"
    if any(word in query for word in ["speaker", "artist", "keynote", "talk", "agenda"]):
        return "speaker"
    if any(word in query for word in ["exhibitor", "expo", "booth", "company"]):
        return "exhibitor"
    if any(word in query for word in ["community", "discord", "slack", "linkedin", "gtm"]):
        return "community"
    if any(word in query for word in ["ops", "operations", "schedule", "conflict", "room"]):
        return "event_ops"
    return "sponsor"


def clear_resource_caches() -> None:
    for resource in (get_embeddings, get_vectordb, get_llm, get_pricing_engine, get_contact_vectordb):
        try:
            resource.clear()
        except Exception:
            pass


def build_monitor_update(
    state: GraphState,
    node: str,
    status: str,
    details: str = "",
    agent: str = "",
    quality_name: str = "",
    quality_verdict: str = "",
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    selected_agent = safe_text(agent or state.get("selected_agent", "UNKNOWN"), "UNKNOWN")

    text_logs = list(state.get("logs", []))
    text_logs.append(f"{timestamp} | {selected_agent} | {node} | {status} | {details}")

    execution_logs = list(state.get("execution_logs", []))
    execution_logs.append(
        {
            "timestamp": timestamp,
            "agent": selected_agent,
            "node": node,
            "status": status,
            "details": details,
        }
    )

    state_flow = list(state.get("state_flow", []))
    state_flow.append(node)

    update: Dict[str, Any] = {
        "logs": text_logs,
        "execution_logs": execution_logs,
        "state_flow": state_flow,
    }

    if quality_name and quality_verdict:
        quality_results = list(state.get("quality_results", []))
        quality_results.append(
            {
                "timestamp": timestamp,
                "check": quality_name,
                "verdict": quality_verdict,
                "agent": selected_agent,
            }
        )
        update["quality_results"] = quality_results

    return update


AGENT_SUBGRAPH_KEYS = {
    "SPONSOR": "sponsor_subgraph",
    "SPEAKER": "speaker_subgraph",
    "EXHIBITOR": "exhibitor_subgraph",
    "VENUE": "venue_subgraph",
    "PRICING": "pricing_subgraph",
    "COMMUNITY": "community_subgraph",
    "EVENT_OPS": "event_ops_subgraph",
    "EMAIL_OUTREACH": "email_outgraph",
}

AGENT_ALIAS_TO_CANONICAL = {
    "sponsor": "SPONSOR",
    "sponsors": "SPONSOR",
    "speaker": "SPEAKER",
    "speakers": "SPEAKER",
    "exhibitor": "EXHIBITOR",
    "exhibitors": "EXHIBITOR",
    "venue": "VENUE",
    "pricing": "PRICING",
    "price": "PRICING",
    "community": "COMMUNITY",
    "event_ops": "EVENT_OPS",
    "event ops": "EVENT_OPS",
    "ops": "EVENT_OPS",
    "email": "EMAIL_OUTREACH",
    "outreach": "EMAIL_OUTREACH",
    "email_outreach": "EMAIL_OUTREACH",
}


def get_agent_subgraphs() -> Dict[str, Any]:
    return {
        "SPONSOR": build_sponsor_subgraph(),
        "SPEAKER": build_speaker_subgraph(),
        "EXHIBITOR": build_exhibitor_subgraph(),
        "VENUE": build_venue_subgraph(),
        "PRICING": build_pricing_subgraph(),
        "COMMUNITY": build_community_subgraph(),
        "EVENT_OPS": build_event_ops_subgraph(),
        "EMAIL_OUTREACH": build_email_subgraph(),
    }


def _canonical_agent_name(agent_name: str) -> str:
    normalized = normalize_whitespace(agent_name).lower()
    return AGENT_ALIAS_TO_CANONICAL.get(normalized, normalized.upper())


def _parse_jsonish(text: str) -> Dict[str, Any]:
    raw_text = safe_text(text)
    if not raw_text:
        return {}
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _parse_agent_list(text: str) -> List[str]:
    parsed = _parse_jsonish(text)
    agents = parsed.get("agents") or parsed.get("agent") or []
    if isinstance(agents, str):
        agents = [agents]
    if not isinstance(agents, list):
        return []
    normalized = []
    for agent in agents:
        canonical = _canonical_agent_name(str(agent))
        if canonical in AGENT_SUBGRAPH_KEYS and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _parse_orchestration_plan(text: str) -> Dict[str, Any]:
    parsed = _parse_jsonish(text)
    agents = _parse_agent_list(json.dumps(parsed, ensure_ascii=True)) if parsed else []
    handoffs = parsed.get("handoffs", []) if isinstance(parsed.get("handoffs", []), list) else []
    normalized_handoffs: List[Dict[str, str]] = []
    for handoff in handoffs:
        if not isinstance(handoff, dict):
            continue
        from_agent = _canonical_agent_name(str(handoff.get("from", "")))
        to_agent = _canonical_agent_name(str(handoff.get("to", "")))
        if from_agent in AGENT_SUBGRAPH_KEYS and to_agent in AGENT_SUBGRAPH_KEYS:
            normalized_handoffs.append({
                "from": from_agent,
                "to": to_agent,
                "reason": safe_text(handoff.get("reason"), ""),
            })
    return {
        "agents": agents,
        "handoffs": normalized_handoffs,
        "reasoning": safe_text(parsed.get("reasoning"), safe_text(parsed.get("reason"), "")),
    }


def _apply_dependency_order(agents: List[str], handoffs: List[Dict[str, str]]) -> List[str]:
    ordered_unique = []
    for agent in agents:
        if agent in AGENT_SUBGRAPH_KEYS and agent not in ordered_unique:
            ordered_unique.append(agent)
    if not ordered_unique:
        return []

    edges = {agent: set() for agent in ordered_unique}
    incoming = {agent: set() for agent in ordered_unique}
    for handoff in handoffs:
        left = handoff.get("from")
        right = handoff.get("to")
        if left in edges and right in edges and left != right:
            edges[left].add(right)
            incoming[right].add(left)

    queue = [agent for agent in ordered_unique if not incoming[agent]]
    resolved: List[str] = []
    while queue:
        current = queue.pop(0)
        resolved.append(current)
        for neighbor in list(edges[current]):
            incoming[neighbor].discard(current)
            if not incoming[neighbor] and neighbor not in resolved and neighbor not in queue:
                queue.append(neighbor)

    for agent in ordered_unique:
        if agent not in resolved:
            resolved.append(agent)
    return resolved


def _planner_agents_with_llm(query_text: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are a planning coordinator for event-planning agents.\n\n"
        "Select the smallest useful set of agents from this exact list only: sponsor, speaker, exhibitor, venue, pricing, community, event_ops, email_outreach.\n"
        "For complex cross-functional requests, choose multiple agents and include dependency handoffs.\n"
        "Return STRICT JSON only with keys: agents (list), handoffs (list of {from, to, reason}), reasoning (string).\n\n"
        f"User request: {query_text}\n"
        f"Event details: {json.dumps(user_input, ensure_ascii=True)}\n"
    )
    raw = call_llm(prompt, temperature=0.0)
    plan = _parse_orchestration_plan(raw)
    if not plan.get("agents"):
        fallback_agent = _canonical_agent_name(choose_agent(user_input))
        plan = {
            "agents": [fallback_agent],
            "handoffs": [],
            "reasoning": "Fallback plan based on intent heuristics.",
        }
    ordered_agents = _apply_dependency_order(plan.get("agents", []), plan.get("handoffs", []))
    plan["agents"] = ordered_agents or plan.get("agents", [])
    return plan


def _run_agent_subgraph(agent: str, state: GraphState) -> Dict[str, Any]:
    subgraphs = get_agent_subgraphs()
    selected_agent = _canonical_agent_name(agent)
    graph = subgraphs.get(selected_agent)
    if graph is None:
        return {"sponsors_answer": f"Unsupported agent: {agent}", "selected_agent": selected_agent}

    invoke_state = dict(state)
    invoke_state["selected_agent"] = selected_agent
    invoke_state["route_target"] = AGENT_SUBGRAPH_KEYS[selected_agent]
    invoke_state["required_agents"] = list(state.get("required_agents", []))
    invoke_state["agent_sequence"] = list(state.get("agent_sequence", []))
    invoke_state["orchestration_plan"] = dict(state.get("orchestration_plan", {}))
    invoke_state["shared_context"] = safe_text(state.get("shared_context", ""))
    invoke_state["context"] = safe_text(state.get("shared_context", "") or state.get("context", ""))

    result = graph.invoke(invoke_state)
    return result


def coordinator_node(state: GraphState) -> dict:
    ui = state.get("user_input", {})
    query_text = safe_text(state.get("query", ui.get("query", "")))
    plan = dict(state.get("orchestration_plan", {}))
    if not plan.get("agents"):
        plan = _planner_agents_with_llm(query_text, ui)

    ordered_agents = _apply_dependency_order(plan.get("agents", []), plan.get("handoffs", []))
    if not ordered_agents:
        ordered_agents = ["SPONSOR"]
        plan["agents"] = ordered_agents
        plan.setdefault("handoffs", [])
        plan.setdefault("reasoning", "Fallback single-agent coordination plan.")

    shared_context = safe_text(state.get("shared_context", ""))
    agent_outputs = dict(state.get("agent_outputs", {}))
    logs = list(state.get("logs", []))
    execution_logs = list(state.get("execution_logs", []))
    state_flow = list(state.get("state_flow", []))
    quality_results = list(state.get("quality_results", []))

    logs.append(f"coordinator: planning {len(ordered_agents)} agent(s)")
    state_flow.append("coordinator_node")
    execution_logs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "COORDINATOR",
        "node": "coordinator_node",
        "status": "planning",
        "details": f"agents={ordered_agents}",
    })

    for agent in ordered_agents:
        agent_state = dict(state)
        agent_state["selected_agent"] = agent
        agent_state["route_target"] = AGENT_SUBGRAPH_KEYS.get(agent, "sponsor_subgraph")
        agent_state["required_agents"] = ordered_agents
        agent_state["agent_sequence"] = ordered_agents
        agent_state["orchestration_plan"] = plan
        agent_state["shared_context"] = shared_context
        agent_state["context"] = "\n\n".join(part for part in [safe_text(state.get("context", "")), shared_context] if part)
        agent_state["agent_outputs"] = agent_outputs
        agent_state["logs"] = logs
        agent_state["execution_logs"] = execution_logs
        agent_state["state_flow"] = state_flow
        agent_state["quality_results"] = quality_results

        result = _run_agent_subgraph(agent, agent_state)
        answer = safe_text(result.get("final_answer") or result.get("sponsors_answer") or result.get("answer") or "")
        agent_outputs[agent] = {
            "answer": answer,
            "context": result.get("context", ""),
            "pricing": result.get("pricing", {}),
            "emails": result.get("emails", []),
            "email_logs": result.get("email_logs", []),
            "logs": result.get("logs", []),
            "execution_logs": result.get("execution_logs", []),
        }
        shared_context = "\n\n".join(part for part in [shared_context, f"[{agent}]\n{answer}"] if part)
        logs.append(f"coordinator: completed {agent}")
        if result.get("logs"):
            logs.extend(result.get("logs", []))
        if result.get("execution_logs"):
            execution_logs.extend(result.get("execution_logs", []))
        if result.get("state_flow"):
            state_flow.extend(result.get("state_flow", []))
        if result.get("quality_results"):
            quality_results.extend(result.get("quality_results", []))

    monitor_update = build_monitor_update(
        state,
        node="coordinator_node",
        status="completed",
        details=f"agents={','.join(ordered_agents)}",
        agent="COORDINATOR",
    )
    logs = monitor_update.get("logs", logs)
    execution_logs = monitor_update.get("execution_logs", execution_logs)
    state_flow = monitor_update.get("state_flow", state_flow)

    return {
        "selected_agent": "COORDINATOR",
        "route_target": "coordinator_node",
        "required_agents": ordered_agents,
        "agent_sequence": ordered_agents,
        "orchestration_plan": plan,
        "shared_context": shared_context,
        "agent_outputs": agent_outputs,
        "logs": logs,
        "execution_logs": execution_logs,
        "state_flow": state_flow,
        "quality_results": quality_results,
    }


def combine_results(state: GraphState) -> dict:
    agent_outputs = dict(state.get("agent_outputs", {}))
    ordered_agents = list(state.get("agent_sequence", [])) or list(agent_outputs.keys())
    plan = state.get("orchestration_plan", {})
    context = "\n\n".join(
        [
            f"Agent sequence: {', '.join(ordered_agents)}",
            f"Orchestration plan: {json.dumps(plan, ensure_ascii=True)}",
            f"Shared context:\n{state.get('shared_context', '')}",
            "Agent outputs:",
            json.dumps(agent_outputs, ensure_ascii=True, indent=2),
        ]
    )
    prompt = (
        "You are a synthesis engine for multi-agent event planning.\n\n"
        f"User request: {json.dumps(state.get('user_input', {}), ensure_ascii=True)}\n\n"
        f"Context to synthesize:\n{context}\n\n"
        "Task:\n"
        "Create one integrated, actionable response that resolves overlaps between agents, highlights dependencies, and produces next steps.\n"
        "If one agent is enough, preserve its recommendation. If multiple agents ran, weave them into one coherent plan.\n"
        "Return a concise markdown answer."
    )
    final_answer = call_llm(prompt)
    monitor_update = build_monitor_update(
        state,
        node="combine_results",
        status="completed",
        details=f"agents={len(agent_outputs)}",
        agent="COORDINATOR",
    )
    return {
        "final_answer": final_answer,
        "sponsors_answer": final_answer,
        "answer": final_answer,
        "logs": list(state.get("logs", [])) + monitor_update.get("logs", []),
        "execution_logs": list(state.get("execution_logs", [])) + monitor_update.get("execution_logs", []),
        "state_flow": list(state.get("state_flow", [])) + monitor_update.get("state_flow", []),
        "quality_results": list(state.get("quality_results", [])) + monitor_update.get("quality_results", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def router_node(state: GraphState) -> dict:
    """Planner-router — decides whether to run one agent or coordinate several."""
    ui = state["user_input"]
    query_text = safe_text(ui.get("query", state.get("query", "")))
    forced_agent = _canonical_agent_name(state.get("selected_agent", ""))

    if forced_agent:
        orchestration_plan = {"mode": "single", "agents": [forced_agent], "reason": "forced"}
        required_agents = [forced_agent]
        selected_agent = forced_agent
    else:
        orchestration_plan = _planner_agents_with_llm(query_text, ui)
        required_agents = orchestration_plan.get("agents", [])
        selected_agent = required_agents[0] if required_agents else _canonical_agent_name(choose_agent(ui))

    if len(required_agents) > 1:
        route_target = "coordinator_node"
        route_label = "COORDINATOR"
    else:
        route_target = AGENT_SUBGRAPH_KEYS.get(selected_agent, "sponsor_subgraph")
        route_label = selected_agent

    print(f"[router_node] selected -> {route_label} | route_target={route_target} | agents={required_agents}")
    monitor_update = build_monitor_update(
        state,
        node="router",
        status="selected",
        details=f"route_target={route_target}; agents={required_agents}",
        agent=route_label,
    )
    return {
        "selected_agent": route_label,
        "route_target": route_target,
        "required_agents": required_agents,
        "agent_sequence": required_agents,
        "orchestration_plan": orchestration_plan,
        "shared_context": "",
        "agent_outputs": {},
        **monitor_update,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SPONSOR AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_sponsor_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"{ui.get('category', 'conference')} conference sponsors in {ui.get('location', 'global')} "
        f"audience size {ui.get('audience_size', 'large')} "
        f"budget {ui.get('budget', 'medium')} "
        "recent sponsorships in last 12 months and marketing spend signals"
    )
    print(f"[build_sponsor_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_sponsor(state: GraphState) -> dict:
    results = retrieve_from_db("sponsor", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_sponsor] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_sponsor_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether a sponsor document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_sponsor_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_sponsor_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    category = ui.get("category", "conference")
    location = ui.get("location", "global")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " is a "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} sponsorship {category} {location} recent news"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "sponsor",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("sponsor") + joined.count("partnership") + joined.count("marketing"),
        })
    print(f"[enrich_sponsor_with_web] sponsor web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_sponsor(state: GraphState) -> dict:
    prompt = (
        "You are a sponsorship strategist.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal sponsor context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend top sponsors and output strict markdown tables:\n"
        "1) Sponsor Prioritization Table\n"
        "| Rank | Company | Why Relevant | Past Sponsorships | Budget Signal | Priority Score (/100) |\n"
        "2) Custom Sponsorship Proposal Table\n"
        "| Package | Benefits | Price | Target Companies |"
    )
    answer = call_llm(prompt)
    print("[generate_sponsor] sponsor answer generated")
    monitor_update = build_monitor_update(
        state,
        node="sponsor.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="SPONSOR",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# SPEAKER AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_speaker_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"{ui.get('event_topic', 'technology')} keynote speakers artists in {ui.get('location', 'global')} "
        f"audience size {ui.get('audience_size', 'large')} "
        f"budget {ui.get('budget', 'medium')} "
        "thought leaders influencers recent conferences publications"
    )
    print(f"[build_speaker_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_speaker(state: GraphState) -> dict:
    results = retrieve_from_db("speaker", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_speaker] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_speaker_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether a speaker document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_speaker_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_speaker_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    topic = ui.get("event_topic", "technology")
    location = ui.get("location", "global")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " is a "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} speaker keynote {topic} {location} recent talks publications"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "speaker",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("speaker") + joined.count("keynote") + joined.count("conference") + joined.count("published"),
        })
    print(f"[enrich_speaker_with_web] speaker web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_speaker(state: GraphState) -> dict:
    prompt = (
        "You are a programming strategist and speaker coordinator.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal speaker context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend top speakers/artists and output strict markdown tables:\n"
        "1) Speaker Prioritization Table\n"
        "| Rank | Name | Expertise | Past Conferences | Speaking Fee | Priority Score (/100) |\n"
        "2) Agenda Mapping Table\n"
        "| Speaker | Session Title | Duration | Audience Fit |\n"
    )
    answer = call_llm(prompt)
    print("[generate_speaker] speaker answer generated")
    monitor_update = build_monitor_update(
        state,
        node="speaker.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="SPEAKER",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# EXHIBITOR AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_exhibitor_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"companies exhibitors that exhibited at {ui.get('category', 'conference')} conferences "
        f"in {ui.get('location', 'global')} "
        f"audience size {ui.get('audience_size', 'large')} "
        f"budget {ui.get('budget', 'medium')} "
        "startup enterprise technology product tools"
    )
    print(f"[build_exhibitor_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_exhibitor(state: GraphState) -> dict:
    results = retrieve_from_db("exhibitor", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_exhibitor] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_exhibitor_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether an exhibitor document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_exhibitor_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_exhibitor_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    category = ui.get("category", "technology")
    location = ui.get("location", "global")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " exhibited at "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} exhibited {category} conference booth product innovation"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "exhibitor",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("exhibitor") + joined.count("booth") + joined.count("conference"),
        })
    print(f"[enrich_exhibitor_with_web] exhibitor web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_exhibitor(state: GraphState) -> dict:
    prompt = (
        "You are an exhibition planning strategist.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal exhibitor context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend top exhibitors and output strict markdown tables:\n"
        "1) Exhibitor Recommendation Table\n"
        "| Rank | Company | Industry | Booth Size | Expected Traffic | Priority (/100) |\n"
        "2) Exhibitor Cluster Table\n"
        "| Cluster | Companies | Synergy | Hall Zone |\n"
    )
    answer = call_llm(prompt)
    print("[generate_exhibitor] exhibitor answer generated")
    monitor_update = build_monitor_update(
        state,
        node="exhibitor.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="EXHIBITOR",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# VENUE AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_venue_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"venues halls in {ui.get('city', ui.get('location', 'global'))} "
        f"capacity around {ui.get('audience_size', '2000')} "
        f"budget {ui.get('budget', 'medium')} "
        f"for {ui.get('category', 'conference')} event "
        "rental pricing reviews accessibility parking"
    )
    print(f"[build_venue_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_venue(state: GraphState) -> dict:
    results = retrieve_from_db("venue", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_venue] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_venue_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether a venue document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_venue_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_venue_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    city = ui.get("city", ui.get("location", "global"))
    category = ui.get("category", "conference")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " is located in "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} venue rental capacity {city} events reviews accessibility"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "venue",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("capacity") + joined.count("venue") + joined.count("event"),
        })
    print(f"[enrich_venue_with_web] venue web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_venue(state: GraphState) -> dict:
    prompt = (
        "You are a venue selection strategist.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal venue context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend top venues and output strict markdown tables:\n"
        "1) Venue Recommendation Table\n"
        "| Rank | Venue Name | Capacity | Location | Price | Amenities | Priority (/100) |\n"
        "2) Venue Shortlist Rationale\n"
        "| Venue | Why Best Fit | Layout | Parking | Accessibility |\n"
    )
    answer = call_llm(prompt)
    print("[generate_venue] venue answer generated")
    monitor_update = build_monitor_update(
        state,
        node="venue.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="VENUE",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# COMMUNITY AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_community_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"{ui.get('category', 'technology')} communities forums in {ui.get('location', 'global')} "
        f"discord slack linkedin meetup groups "
        f"audience size {ui.get('audience_size', 'large')} "
        f"event promotion GTM distribution engagement"
    )
    print(f"[build_community_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_community(state: GraphState) -> dict:
    results = retrieve_from_db("community", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_community] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_community_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether a community document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_community_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_community_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    category = ui.get("category", "technology")
    location = ui.get("location", "global")
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = " is a "
        name = doc.split(marker)[0].strip() if marker in doc else doc[:60].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    web_profiles = []
    for name in candidates[:8]:
        q = f"{name} community {category} discord slack linkedin event promotion engagement"
        hits = tavily_search(q, max_results=5)
        joined = " ".join(h.get("snippet", "") for h in hits).lower()
        web_profiles.append({
            "name": name,
            "kind": "community",
            "query": q,
            "hits": hits,
            "signal_score": joined.count("community") + joined.count("engagement") + joined.count("members"),
        })
    print(f"[enrich_community_with_web] community web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_community(state: GraphState) -> dict:
    prompt = (
        "You are a GTM strategist for community-driven promotion.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal community context:\n"
        f"{state['context']}\n\n"
        "Tavily web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Recommend communities and GTM strategy with markdown tables:\n"
        "1) Community Prioritization\n"
        "| Rank | Community | Members | Activity | Fit | Priority (/100) |\n"
        "2) Messaging Strategy\n"
        "| Community | Key Message | Best Time | Content Format |\n"
        "3) Distribution Plan\n"
        "| Channel | Frequency | Owner | Budget | Expected Reach |\n"
    )
    answer = call_llm(prompt)
    print("[generate_community] community answer generated")
    monitor_update = build_monitor_update(
        state,
        node="community.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="COMMUNITY",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# EVENT OPS AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_event_ops_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"event agenda scheduling timeline for {ui.get('category', 'conference')} "
        f"audience {ui.get('audience_size', '2000')} "
        f"in {ui.get('city', ui.get('location', 'global'))} "
        "room assignment speaker slots time conflicts resource planning"
    )
    print(f"[build_event_ops_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_event_ops(state: GraphState) -> dict:
    results = retrieve_from_db("event_ops", state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_event_ops] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_event_ops_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are checking whether an event operations document is relevant.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Document: {doc}\n\n"
            "Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
    context = "\n\n".join(relevant)
    print(f"[filter_event_ops_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_event_ops_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    category = ui.get("category", "conference")
    location = ui.get("location", "global")
    q = f"{category} event operations agenda scheduling best practices resource planning {location}"
    hits = tavily_search(q, max_results=6)
    joined = " ".join(h.get("snippet", "") for h in hits).lower()
    web_profiles = [{
        "kind": "event_ops_best_practices",
        "query": q,
        "hits": hits,
        "ops_signal": joined.count("schedule") + joined.count("agenda") + joined.count("resource"),
        "conflict_signal": joined.count("conflict") + joined.count("overlap") + joined.count("room"),
    }]
    print("[enrich_event_ops_with_web] event ops web profiles=1")
    return {"web_profiles": web_profiles}


def generate_event_ops(state: GraphState) -> dict:
    prompt = (
        "You are an event operations and planning expert.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Internal ops context:\n"
        f"{state['context']}\n\n"
        "External best practices:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Build a practical event agenda with markdown tables:\n"
        "1) Event Agenda Timeline\n"
        "| Time | Session/Activity | Room | Speakers | Duration | Capacity |\n"
        "2) Room/Resource Assignment\n"
        "| Room | Max Capacity | Sessions | Equipment | AV Support |\n"
        "3) Potential Conflicts & Mitigations\n"
        "| Conflict | Impact | Mitigation | Owner |\n"
        "Use concise, execution-ready format.\n"
    )
    answer = call_llm(prompt)
    print("[generate_event_ops] event ops answer generated")
    monitor_update = build_monitor_update(
        state,
        node="event_ops.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="EVENT_OPS",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# EMAIL OUTREACH AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_email_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = f"outreach contact list for {ui.get('category', 'conference')} in {ui.get('location', 'global')}"
    print(f"[build_email_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_email_contacts(state: GraphState) -> dict:
    db = get_contact_vectordb()
    if db is None:
        print("[retrieve_email_contacts] No contact DB; returning empty")
        return {"raw_docs": []}
    results = db.similarity_search(state["retrieval_query"], k=20)
    docs = [r.page_content for r in results]
    print(f"[retrieve_email_contacts] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_email_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are filtering contacts for event outreach.\n\n"
            f"Event details: {state['user_input']}\n\n"
            f"Contact: {doc}\n\n"
            "Is this person a good fit for outreach? Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:10]
    context = "\n\n".join(relevant)
    print(f"[filter_email_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": context}


def enrich_email_with_web(state: GraphState) -> dict:
    candidates = []
    seen = set()
    for doc in state["relevant_docs"]:
        marker = "Email: "
        email = ""
        if marker in doc:
            email = doc.split(marker)[1].split(".")[0] if marker in doc else ""
        name = doc.split("works at")[0].strip() if "works at" in doc else doc[:40].strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append((name, email))
    web_profiles = []
    for name, email in candidates[:8]:
        q = f"{name} linkedin profile {email}"
        hits = tavily_search(q, max_results=3)
        web_profiles.append({
            "name": name,
            "email": email,
            "kind": "contact",
            "query": q,
            "hits": hits,
        })
    print(f"[enrich_email_with_web] email web profiles={len(web_profiles)}")
    return {"web_profiles": web_profiles}


def generate_email(state: GraphState) -> dict:
    prompt = (
        "You are a persuasive email outreach strategist.\n\n"
        f"Event details: {state['user_input']}\n\n"
        "Contact list:\n"
        f"{state['context']}\n\n"
        "Web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "Generate personalized outreach email templates. Output strict markdown format:\n"
        "1) Email Template (General)\n"
        "Subject: [subject]\n"
        "Body: [body with personalization placeholders]\n"
        "2) Contact Segmentation\n"
        "| Segment | Count | Key Message | CTA |\n"
        "3) Follow-up Strategy\n"
        "| Day | Action | Channel |\n"
    )
    answer = call_llm(prompt)
    print("[generate_email] email outreach answer generated")
    monitor_update = build_monitor_update(
        state,
        node="email_outreach.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="EMAIL_OUTREACH",
    )
    return {"sponsors_answer": answer, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY CONTROL NODES
# ═══════════════════════════════════════════════════════════════════════════

def check_hallucination(state: GraphState) -> dict:
    prompt = (
        "You are a fact-checker.\n\n"
        "Context (ground truth):\n"
        f"{state.get('context', '')}\n\n"
        "Web evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Answer to verify:\n"
        f"{state.get('sponsors_answer', '')}\n\n"
        "Is the answer fully supported by the context, partially supported, or not supported?\n"
        "Reply ONLY with one of: Fully Supported / Partially Supported / No Support"
    )
    verdict = call_llm(prompt, temperature=0.0)
    print(f"[check_hallucination] verdict: {verdict}")
    monitor_update = build_monitor_update(
        state,
        node="quality.hallucination",
        status="completed",
        details=verdict,
        quality_name="hallucination",
        quality_verdict=verdict,
    )
    return {"hallucination_verdict": verdict, **monitor_update}


def revise(state: GraphState) -> dict:
    prompt = (
        "The following recommendation contains unsupported claims.\n\n"
        "Context (only use this):\n"
        f"{state.get('context', '')}\n\n"
        "Web evidence (only use this):\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Original answer:\n"
        f"{state.get('sponsors_answer', '')}\n\n"
        "Rewrite using ONLY grounded information and preserve markdown table format."
    )
    revised = call_llm(prompt)
    new_count = state.get("revise_count", 0) + 1
    print(f"[revise] revision #{new_count}")
    monitor_update = build_monitor_update(
        state,
        node="quality.revise",
        status="completed",
        details=f"revise_count={new_count}",
    )
    return {"sponsors_answer": revised, "revise_count": new_count, **monitor_update}


def check_usefulness(state: GraphState) -> dict:
    prompt = (
        "You are evaluating the quality of an event recommendation.\n\n"
        f"Event details: {state.get('user_input', {})}\n\n"
        f"Recommendation:\n{state.get('sponsors_answer', '')}\n\n"
        "Is this recommendation useful and actionable for the organizer? "
        "Reply ONLY with: Useful / Not Useful"
    )
    verdict = call_llm(prompt, temperature=0.0)
    print(f"[check_usefulness] verdict: {verdict}")
    monitor_update = build_monitor_update(
        state,
        node="quality.usefulness",
        status="completed",
        details=verdict,
        quality_name="usefulness",
        quality_verdict=verdict,
    )
    return {"usefulness_verdict": verdict, **monitor_update}


def rewrite_query(state: GraphState) -> dict:
    prompt = (
        "The following query did not retrieve useful information.\n\n"
        f"Original query: {state.get('retrieval_query', '')}\n"
        f"Event details: {state.get('user_input', {})}\n\n"
        "Write an improved, more specific search query. Return ONLY the query string."
    )
    new_query = call_llm(prompt, temperature=0.5)
    new_count = state.get("rewrite_count", 0) + 1
    print(f"[rewrite_query] new query: '{new_query}' (attempt #{new_count})")
    monitor_update = build_monitor_update(
        state,
        node="quality.rewrite_query",
        status="completed",
        details=f"rewrite_count={new_count}",
    )
    return {"retrieval_query": new_query, "rewrite_count": new_count, **monitor_update}


# ═══════════════════════════════════════════════════════════════════════════
# PRICING AGENT NODES
# ═══════════════════════════════════════════════════════════════════════════

def build_pricing_query(state: GraphState) -> dict:
    ui = state["user_input"]
    query = (
        f"ticket pricing tiers and conversion for {ui.get('category', 'conference')} events "
        f"in {ui.get('city', ui.get('location', 'global'))} "
        f"audience {ui.get('audience_size', '2000')} budget {ui.get('budget', 'medium')}"
    )
    print(f"[build_pricing_query] -> '{query}'")
    return {"retrieval_query": query}


def retrieve_pricing(state: GraphState) -> dict:
    engine = get_pricing_engine()
    results = engine.pricing_vectordb.similarity_search(state["retrieval_query"], k=12)
    docs = [r.page_content for r in results]
    print(f"[retrieve_pricing] docs={len(docs)}")
    return {"raw_docs": docs}


def filter_pricing_relevance(state: GraphState) -> dict:
    relevant = []
    for doc in state["raw_docs"]:
        prompt = (
            "You are evaluating historical pricing evidence relevance.\n\n"
            f"Target event details: {state.get('user_input', {})}\n\n"
            f"Historical record: {doc}\n\n"
            "Is this relevant for ticket pricing and footfall forecasting? Reply ONLY with YES or NO."
        )
        verdict = call_llm(prompt, temperature=0.0)
        if "YES" in verdict.upper():
            relevant.append(doc)
    if not relevant:
        relevant = state["raw_docs"][:6]
        print("[filter_pricing_relevance] fallback -> top 6 raw docs")
    else:
        print(f"[filter_pricing_relevance] relevant={len(relevant)}")
    return {"relevant_docs": relevant, "context": "\n\n".join(relevant)}


def enrich_pricing_with_web(state: GraphState) -> dict:
    ui = state["user_input"]
    city = ui.get("city", ui.get("location", "global"))
    category = ui.get("category", "conference")
    q = (
        f"{category} conference ticket pricing trends {city} conversion rate early bird regular vip "
        "attendance demand signals"
    )
    hits = tavily_search(q, max_results=6)
    joined = " ".join(h.get("snippet", "") for h in hits).lower()
    web_profiles = [{
        "kind": "pricing_market",
        "query": q,
        "hits": hits,
        "pricing_signal": joined.count("price") + joined.count("ticket") + joined.count("pricing"),
        "conversion_signal": joined.count("conversion") + joined.count("sellout") + joined.count("demand"),
        "footfall_signal": joined.count("attendance") + joined.count("footfall") + joined.count("visitors"),
    }]
    print("[enrich_pricing_with_web] pricing web profiles=1")
    return {"web_profiles": web_profiles}


def generate_pricing(state: GraphState) -> dict:
    engine = get_pricing_engine()
    result = engine.run(state["user_input"])
    prompt = (
        "You are a pricing and footfall strategist.\n\n"
        f"Target event details: {state.get('user_input', {})}\n\n"
        "Historical pricing context:\n"
        f"{state.get('context', '')}\n\n"
        "External market evidence:\n"
        f"{json.dumps(state.get('web_profiles', []), ensure_ascii=True, indent=2)}\n\n"
        "Model output:\n"
        f"{json.dumps(result, ensure_ascii=True, indent=2)}\n\n"
        "Task: Produce strict markdown with pricing strategy, forecast summary, and three grounded insights."
    )
    answer = call_llm(prompt)
    print("[generate_pricing] pricing answer generated")
    monitor_update = build_monitor_update(
        state,
        node="pricing.generate",
        status="completed",
        details=f"answer_chars={len(answer)}",
        agent="PRICING",
    )
    return {
        "pricing": result,
        "sponsors_answer": answer,
        "context": state.get("context", ""),
        "web_profiles": state.get("web_profiles", []),
        **monitor_update,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

MAX_REVISIONS = 1
MAX_REWRITES = 1


def route_after_router(state: GraphState) -> str:
    return safe_text(state.get("route_target", "sponsor_subgraph"), "sponsor_subgraph")


def route_after_hallucination_check(state: GraphState) -> str:
    verdict = state.get("hallucination_verdict", "").upper()
    if "FULLY" in verdict:
        return "check_usefulness"
    if state.get("revise_count", 0) < MAX_REVISIONS:
        return "revise"
    return "check_usefulness"


def route_after_usefulness_check(state: GraphState) -> str:
    verdict = state.get("usefulness_verdict", "").upper()
    if "NOT USEFUL" in verdict and state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite_query"
    return END


def route_after_rewrite(state: GraphState) -> str:
    return safe_text(state.get("route_target", "sponsor_subgraph"), "sponsor_subgraph")


# ═══════════════════════════════════════════════════════════════════════════
# SUBGRAPH BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_sponsor_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_sponsor_query)
    sg.add_node("retrieve", retrieve_sponsor)
    sg.add_node("filter_relevance", filter_sponsor_relevance)
    sg.add_node("enrich_web", enrich_sponsor_with_web)
    sg.add_node("generate", generate_sponsor)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_pricing_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_pricing_query)
    sg.add_node("retrieve", retrieve_pricing)
    sg.add_node("filter_relevance", filter_pricing_relevance)
    sg.add_node("enrich_web", enrich_pricing_with_web)
    sg.add_node("generate", generate_pricing)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_speaker_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_speaker_query)
    sg.add_node("retrieve", retrieve_speaker)
    sg.add_node("filter_relevance", filter_speaker_relevance)
    sg.add_node("enrich_web", enrich_speaker_with_web)
    sg.add_node("generate", generate_speaker)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_exhibitor_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_exhibitor_query)
    sg.add_node("retrieve", retrieve_exhibitor)
    sg.add_node("filter_relevance", filter_exhibitor_relevance)
    sg.add_node("enrich_web", enrich_exhibitor_with_web)
    sg.add_node("generate", generate_exhibitor)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_venue_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_venue_query)
    sg.add_node("retrieve", retrieve_venue)
    sg.add_node("filter_relevance", filter_venue_relevance)
    sg.add_node("enrich_web", enrich_venue_with_web)
    sg.add_node("generate", generate_venue)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_community_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_community_query)
    sg.add_node("retrieve", retrieve_community)
    sg.add_node("filter_relevance", filter_community_relevance)
    sg.add_node("enrich_web", enrich_community_with_web)
    sg.add_node("generate", generate_community)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_event_ops_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_event_ops_query)
    sg.add_node("retrieve", retrieve_event_ops)
    sg.add_node("filter_relevance", filter_event_ops_relevance)
    sg.add_node("enrich_web", enrich_event_ops_with_web)
    sg.add_node("generate", generate_event_ops)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


def build_email_subgraph():
    sg = StateGraph(GraphState)
    sg.add_node("build_query", build_email_query)
    sg.add_node("retrieve", retrieve_email_contacts)
    sg.add_node("filter_relevance", filter_email_relevance)
    sg.add_node("enrich_web", enrich_email_with_web)
    sg.add_node("generate", generate_email)
    
    sg.set_entry_point("build_query")
    sg.add_edge("build_query", "retrieve")
    sg.add_edge("retrieve", "filter_relevance")
    sg.add_edge("filter_relevance", "enrich_web")
    sg.add_edge("enrich_web", "generate")
    sg.add_edge("generate", END)
    
    return sg.compile()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_main_graph():
    subgraphs = get_agent_subgraphs()
    sponsor_subgraph = subgraphs["SPONSOR"]
    speaker_subgraph = subgraphs["SPEAKER"]
    exhibitor_subgraph = subgraphs["EXHIBITOR"]
    venue_subgraph = subgraphs["VENUE"]
    pricing_subgraph = subgraphs["PRICING"]
    community_subgraph = subgraphs["COMMUNITY"]
    event_ops_subgraph = subgraphs["EVENT_OPS"]
    email_outgraph = subgraphs["EMAIL_OUTREACH"]
    
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("coordinator_node", coordinator_node)
    builder.add_node("combine_results", combine_results)
    builder.add_node("sponsor_subgraph", sponsor_subgraph)
    builder.add_node("pricing_subgraph", pricing_subgraph)
    builder.add_node("speaker_subgraph", speaker_subgraph)
    builder.add_node("exhibitor_subgraph", exhibitor_subgraph)
    builder.add_node("venue_subgraph", venue_subgraph)
    builder.add_node("community_subgraph", community_subgraph)
    builder.add_node("event_ops_subgraph", event_ops_subgraph)
    builder.add_node("email_outgraph", email_outgraph)
    builder.add_node("check_hallucination", check_hallucination)
    builder.add_node("revise", revise)
    builder.add_node("check_usefulness", check_usefulness)
    builder.add_node("rewrite_query", rewrite_query)
    
    # Entry point
    builder.set_entry_point("router")
    
    # Routing
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "coordinator_node": "coordinator_node",
            "sponsor_subgraph": "sponsor_subgraph",
            "pricing_subgraph": "pricing_subgraph",
            "speaker_subgraph": "speaker_subgraph",
            "exhibitor_subgraph": "exhibitor_subgraph",
            "venue_subgraph": "venue_subgraph",
            "community_subgraph": "community_subgraph",
            "event_ops_subgraph": "event_ops_subgraph",
            "email_outgraph": "email_outgraph",
        }
    )
    
    # Multi-agent synthesis path
    builder.add_edge("coordinator_node", "combine_results")
    builder.add_edge("combine_results", "check_hallucination")
    
    # Quality checks
    builder.add_edge("sponsor_subgraph", "check_hallucination")
    builder.add_edge("pricing_subgraph", "check_hallucination")
    builder.add_edge("speaker_subgraph", "check_hallucination")
    builder.add_edge("exhibitor_subgraph", "check_hallucination")
    builder.add_edge("venue_subgraph", "check_hallucination")
    builder.add_edge("community_subgraph", "check_hallucination")
    builder.add_edge("event_ops_subgraph", "check_hallucination")
    builder.add_edge("email_outgraph", "check_hallucination")
    builder.add_edge("revise", "check_hallucination")
    
    builder.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "check_usefulness": "check_usefulness",
            "revise": "revise",
        }
    )
    
    builder.add_conditional_edges(
        "check_usefulness",
        route_after_usefulness_check,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        }
    )
    
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "coordinator_node": "coordinator_node",
            "sponsor_subgraph": "sponsor_subgraph",
            "pricing_subgraph": "pricing_subgraph",
            "speaker_subgraph": "speaker_subgraph",
            "exhibitor_subgraph": "exhibitor_subgraph",
            "venue_subgraph": "venue_subgraph",
            "community_subgraph": "community_subgraph",
            "event_ops_subgraph": "event_ops_subgraph",
            "email_outgraph": "email_outgraph",
        }
    )
    
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE GRAPH
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_graph():
    return build_main_graph()


# ═══════════════════════════════════════════════════════════════════════════
# PRICING/SIMULATOR FUNCTIONS (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def build_tier_rows(pricing_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not pricing_payload or not isinstance(pricing_payload, dict):
        pricing_payload = {}
    tiers = pricing_payload.get("tiers", {}) or {}
    rows: List[Dict[str, Any]] = []
    if isinstance(tiers, dict):
        iterable = tiers.items()
    elif isinstance(tiers, list):
        iterable = [(tier.get("id", f"tier_{idx}"), tier) for idx, tier in enumerate(tiers)]
    else:
        iterable = []
    for tier_id, tier in iterable:
        if not isinstance(tier, dict):
            continue
        label = safe_text(tier.get("label") or tier_id.replace("_", " ").title(), "Tier")
        price = float(tier.get("price", 0) or 0)
        seats = int(round(tier.get("expected_tickets_sold", tier.get("seats", tier.get("seat_count", 0))) or 0))
        conv = float(tier.get("expected_conversion", tier.get("conv", tier.get("conversion", 0))) or 0)
        revenue = float(tier.get("expected_revenue", price * seats) or price * seats)
        rows.append({
            "id": safe_text(tier.get("id") or tier_id),
            "label": label,
            "price": price,
            "seats": seats,
            "conv": conv,
            "revenue": revenue,
        })
    if not rows:
        rows = [
            {"id": "eb", "label": "Early Bird", "price": 999.0, "seats": 200, "conv": 0.72, "revenue": 143856.0},
            {"id": "gen", "label": "General", "price": 1999.0, "seats": 400, "conv": 0.55, "revenue": 439780.0},
            {"id": "vip", "label": "VIP", "price": 4999.0, "seats": 80, "conv": 0.38, "revenue": 151969.6},
            {"id": "grp", "label": "Group", "price": 1499.0, "seats": 120, "conv": 0.62, "revenue": 111523.2},
        ]
    return rows


def tier_metrics(rows: List[Dict[str, Any]], promo_discount: float) -> Dict[str, Any]:
    gross = sum(float(row["price"]) * float(row["seats"]) * float(row["conv"]) for row in rows)
    total_seats = sum(int(row["seats"]) for row in rows)
    avg_conv = sum(float(row["conv"]) for row in rows) / max(len(rows), 1)
    promo_factor = max(0.0, 1.0 - (promo_discount / 100.0))
    after_promo = gross * promo_factor
    weighted_avg_price = sum(float(row["price"]) * float(row["seats"]) for row in rows) / max(total_seats, 1)
    return {
        "gross_revenue": gross,
        "after_promo_revenue": after_promo,
        "total_seats": total_seats,
        "avg_conversion": avg_conv,
        "weighted_avg_price": weighted_avg_price,
        "promo_factor": promo_factor,
    }


def compute_break_even(
    venue_cost: float,
    speaker_fees: float,
    ops_cost: float,
    marketing_cost: float,
    avg_ticket_price: float,
    total_capacity: int,
    avg_daily_sales: float,
    event_date: datetime,
) -> Dict[str, Any]:
    total_fixed_costs = max(0.0, venue_cost) + max(0.0, speaker_fees) + max(0.0, ops_cost) + max(0.0, marketing_cost)
    avg_ticket_price = max(0.01, avg_ticket_price)
    total_capacity = max(1, int(total_capacity))
    avg_daily_sales = max(0.01, avg_daily_sales)
    breakeven_tickets = int(-(-total_fixed_costs // avg_ticket_price))
    breakeven_pct = round((breakeven_tickets / total_capacity) * 100, 1)
    full_capacity_revenue = total_capacity * avg_ticket_price
    net_profit_at_capacity = full_capacity_revenue - total_fixed_costs
    margin_at_capacity_pct = round((net_profit_at_capacity / full_capacity_revenue) * 100, 1) if full_capacity_revenue > 0 else 0.0
    days_to_breakeven = int(-(-breakeven_tickets // avg_daily_sales))
    if isinstance(event_date, datetime):
        event_day = event_date.date()
    else:
        event_day = event_date
    today = datetime.now(timezone.utc).date()
    days_until_event = (event_day - today).days
    breakeven_date = today + timedelta(days=days_to_breakeven)
    return {
        "total_fixed_costs": total_fixed_costs,
        "breakeven_tickets": breakeven_tickets,
        "breakeven_pct_of_capacity": breakeven_pct,
        "days_to_breakeven": days_to_breakeven,
        "breakeven_date": breakeven_date.isoformat(),
        "days_until_event": days_until_event,
        "on_track": days_to_breakeven <= days_until_event,
        "feasible": breakeven_tickets <= total_capacity,
        "full_capacity_revenue": full_capacity_revenue,
        "net_profit_at_capacity": net_profit_at_capacity,
        "margin_at_capacity_pct": margin_at_capacity_pct,
        "cost_breakdown": {
            "venue": venue_cost,
            "speakers": speaker_fees,
            "ops": ops_cost,
            "marketing": marketing_cost,
        },
    }


def metrics_default_price(tier_rows: List[Dict[str, Any]]) -> float:
    if not tier_rows:
        return 0.0
    numerator = sum(float(row["price"]) * float(row["seats"]) for row in tier_rows)
    denominator = sum(float(row["seats"]) for row in tier_rows) or 1.0
    return numerator / denominator


def render_pricing_simulator(pricing_result: Dict[str, Any]) -> None:
    if not pricing_result or not isinstance(pricing_result, dict):
        st.info("Run the pricing engine first so the simulator can use real tier data.")
        return
    st.subheader("Ticket Tier Simulator")
    tiers = build_tier_rows(pricing_result)
    if not tiers:
        st.info("Run the pricing engine first so the simulator can use real tier data.")
        return
    if "sim_tiers" not in st.session_state:
        st.session_state["sim_tiers"] = [dict(row) for row in tiers]
    if st.button("Reset simulator", key="reset_simulator"):
        st.session_state["sim_tiers"] = [dict(row) for row in tiers]
    promo_discount = st.slider("Promo discount", min_value=0, max_value=50, value=10, step=1, key="promo_discount")
    sim_rows = st.session_state["sim_tiers"]
    for idx, row in enumerate(sim_rows):
        with st.expander(f"{row['label']} tier", expanded=idx == 0):
            c1, c2, c3 = st.columns(3)
            row["price"] = c1.slider(f"{row['label']} price", min_value=100, max_value=25000, value=int(round(row["price"])), step=50, key=f"price_{row['id']}")
            row["seats"] = c2.slider(f"{row['label']} seats", min_value=0, max_value=5000, value=int(row["seats"]), step=10, key=f"seats_{row['id']}")
            row["conv"] = c3.slider(f"{row['label']} conversion", min_value=0.0, max_value=1.0, value=float(max(0.0, min(1.0, row["conv"]))), step=0.01, format="%.2f", key=f"conv_{row['id']}")
    metrics = tier_metrics(sim_rows, promo_discount)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Gross revenue", f"₹{metrics['gross_revenue']:,.0f}")
    metric_cols[1].metric(f"After {promo_discount}% promo", f"₹{metrics['after_promo_revenue']:,.0f}")
    metric_cols[2].metric("Total seats", f"{metrics['total_seats']:,}")
    metric_cols[3].metric("Avg conversion", f"{metrics['avg_conversion'] * 100:.1f}%")
    tier_chart_df = pd.DataFrame({
        "Tier": [row["label"] for row in sim_rows],
        "Gross revenue": [float(row["price"]) * float(row["seats"]) * float(row["conv"]) for row in sim_rows],
        "Promo-adjusted revenue": [float(row["price"]) * float(row["seats"]) * float(row["conv"]) * (1.0 - promo_discount / 100.0) for row in sim_rows],
    }).set_index("Tier")
    st.bar_chart(tier_chart_df)
    summary_rows = []
    for row in sim_rows:
        gross = float(row["price"]) * float(row["seats"]) * float(row["conv"])
        summary_rows.append({
            "Tier": row["label"],
            "Price": f"₹{row['price']:,.0f}",
            "Seats": int(row["seats"]),
            "Conv": f"{row['conv'] * 100:.1f}%",
            "Gross Revenue": f"₹{gross:,.0f}",
            "Promo Revenue": f"₹{gross * (1.0 - promo_discount / 100.0):,.0f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def render_break_even_analysis(pricing_result: Dict[str, Any], user_input: Dict[str, Any]) -> None:
    st.subheader("Break-even analysis")
    if not pricing_result or not isinstance(pricing_result, dict):
        st.info("Run the pricing engine first so break-even analysis can use the ticket price.")
        return
    pricing_payload = pricing_result.get("pricing", pricing_result) if pricing_result else {}
    if not pricing_payload or not isinstance(pricing_payload, dict):
        pricing_payload = {}
    tier_rows = build_tier_rows(pricing_payload)
    default_price = pricing_payload.get("base_price") or metrics_default_price(tier_rows)
    if not default_price:
        st.info("Run the pricing engine first so break-even analysis can use the ticket price.")
        return
    default_capacity = int(pricing_payload.get("venue_capacity") or user_input.get("audience_size") or 1000)
    if "be_state" not in st.session_state:
        st.session_state["be_state"] = {
            "venue_cost": 0.0,
            "speaker_fees": 0.0,
            "ops_cost": 0.0,
            "marketing_cost": 0.0,
            "avg_ticket_price": float(default_price),
            "total_capacity": default_capacity,
            "avg_daily_sales": 1.0,
            "event_date": datetime.now(timezone.utc).date(),
        }
    state = st.session_state["be_state"]
    rows = st.columns(2)
    with rows[0]:
        state["venue_cost"] = st.slider("Venue cost", 0, 5000000, int(state["venue_cost"]), 1000)
        state["speaker_fees"] = st.slider("Speaker fees", 0, 3000000, int(state["speaker_fees"]), 1000)
        state["ops_cost"] = st.slider("Ops cost", 0, 2000000, int(state["ops_cost"]), 1000)
        state["marketing_cost"] = st.slider("Marketing budget", 0, 2000000, int(state["marketing_cost"]), 1000)
    with rows[1]:
        state["avg_ticket_price"] = st.slider("Average ticket price", 100, 25000, int(state["avg_ticket_price"]), 50)
        state["total_capacity"] = st.slider("Capacity", 50, 50000, int(state["total_capacity"]), 50)
        state["avg_daily_sales"] = st.slider("Average daily sales", 1, 1000, int(state["avg_daily_sales"]), 1)
        state["event_date"] = st.date_input("Event date", value=state["event_date"])
    be = compute_break_even(
        venue_cost=float(state["venue_cost"]),
        speaker_fees=float(state["speaker_fees"]),
        ops_cost=float(state["ops_cost"]),
        marketing_cost=float(state["marketing_cost"]),
        avg_ticket_price=float(state["avg_ticket_price"]),
        total_capacity=int(state["total_capacity"]),
        avg_daily_sales=float(state["avg_daily_sales"]),
        event_date=state["event_date"],
    )
    banner = "On track" if be["on_track"] else "At risk"
    banner_color = "green" if be["feasible"] and be["on_track"] else "orange" if be["feasible"] else "red"
    st.markdown(
        f"<div style='padding:0.85rem 1rem;border-radius:14px;background:rgba(255,255,255,0.7);border:1px solid rgba(0,0,0,0.08);color:{banner_color};'><strong>{banner}</strong> - break-even at {be['breakeven_tickets']:,} tickets ({be['breakeven_pct_of_capacity']}% of capacity).</div>",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Break-even tickets", f"{be['breakeven_tickets']:,}", f"{be['breakeven_pct_of_capacity']}% of capacity")
    metric_cols[1].metric("Days to break-even", f"{be['days_to_breakeven']:,}", f"{be['days_until_event']:,} days until event")
    metric_cols[2].metric("Margin at capacity", f"{be['margin_at_capacity_pct']:.1f}%", f"₹{be['net_profit_at_capacity']:,.0f} profit")
    metric_cols[3].metric("Fixed costs", f"₹{be['total_fixed_costs']:,.0f}")
    curve_points = list(range(0, int(state["total_capacity"]) + 1, max(1, int(state["total_capacity"]) // 20)))
    if curve_points[-1] != int(state["total_capacity"]):
        curve_points.append(int(state["total_capacity"]))
    curve_df = pd.DataFrame({
        "Tickets sold": curve_points,
        "Revenue": [pt * float(state["avg_ticket_price"]) for pt in curve_points],
        "Fixed costs": [be["total_fixed_costs"] for _ in curve_points],
    }).set_index("Tickets sold")
    st.line_chart(curve_df)
    st.dataframe(pd.DataFrame([{
        "venue": be["cost_breakdown"]["venue"],
        "speakers": be["cost_breakdown"]["speakers"],
        "ops": be["cost_breakdown"]["ops"],
        "marketing": be["cost_breakdown"]["marketing"],
        "break_even_date": be["breakeven_date"],
        "feasible": be["feasible"],
    }]), use_container_width=True, hide_index=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_forecast_baseline(pricing_result: Dict[str, Any], user_input: Dict[str, Any]) -> Dict[str, Any]:
    if not pricing_result or not isinstance(pricing_result, dict):
        pricing_result = {}
    pricing_payload = pricing_result.get("pricing", pricing_result) if pricing_result else {}
    if not pricing_payload or not isinstance(pricing_payload, dict):
        pricing_payload = {}

    tier_rows = build_tier_rows(pricing_payload)
    base_metrics = tier_metrics(tier_rows, promo_discount=0.0)
    total_tickets = sum(float(row["seats"]) * float(row["conv"]) for row in tier_rows)
    venue_capacity = pricing_payload.get("venue_capacity") or user_input.get("audience_size") or sum(int(row["seats"]) for row in tier_rows)
    conversion_values = [float(row["conv"]) for row in tier_rows] or [0.0]
    avg_conversion = sum(conversion_values) / max(len(conversion_values), 1)
    avg_price = metrics_default_price(tier_rows)
    return {
        "tier_rows": tier_rows,
        "base_revenue": float(base_metrics["gross_revenue"]),
        "base_tickets": float(total_tickets),
        "base_capacity": float(max(1, int(venue_capacity))),
        "base_conversion": float(avg_conversion),
        "base_avg_price": float(avg_price),
    }


def derive_dynamic_forecast_ranges(baseline: Dict[str, Any]) -> Dict[str, float]:
    rows = baseline.get("tier_rows", [])
    if not rows:
        return {
            "price_shift_abs": 15.0,
            "conv_shift_abs": 20.0,
            "demand_shift_abs": 25.0,
            "promo_max": 30.0,
        }

    prices = [float(r["price"]) for r in rows if float(r.get("price", 0) or 0) > 0]
    conversions = [float(r["conv"]) for r in rows]
    seats = [float(r["seats"]) for r in rows]

    avg_price = sum(prices) / max(len(prices), 1)
    price_spread = (max(prices) - min(prices)) if prices else 0.0
    price_shift_abs = clamp((price_spread / max(avg_price, 1.0)) * 100.0, 8.0, 45.0)

    avg_conv = sum(conversions) / max(len(conversions), 1)
    conv_spread = (max(conversions) - min(conversions)) if conversions else 0.0
    conv_shift_abs = clamp((conv_spread / max(avg_conv, 0.01)) * 100.0, 10.0, 60.0)

    avg_seat = sum(seats) / max(len(seats), 1)
    seat_spread = (max(seats) - min(seats)) if seats else 0.0
    demand_shift_abs = clamp((seat_spread / max(avg_seat, 1.0)) * 100.0, 12.0, 65.0)

    promo_max = clamp(conv_shift_abs, 15.0, 50.0)
    return {
        "price_shift_abs": float(round(price_shift_abs, 1)),
        "conv_shift_abs": float(round(conv_shift_abs, 1)),
        "demand_shift_abs": float(round(demand_shift_abs, 1)),
        "promo_max": float(round(promo_max, 1)),
    }


def apply_pricing_scenario(
    tier_rows: List[Dict[str, Any]],
    price_shift_pct: float,
    conversion_shift_pct: float,
    demand_shift_pct: float,
    promo_discount_pct: float,
) -> Dict[str, Any]:
    scenario_rows: List[Dict[str, Any]] = []
    for row in tier_rows:
        scenario_price = max(0.0, float(row["price"]) * (1.0 + price_shift_pct / 100.0))
        scenario_conv = clamp(float(row["conv"]) * (1.0 + conversion_shift_pct / 100.0), 0.0, 1.0)
        scenario_seats = max(0, int(round(float(row["seats"]) * (1.0 + demand_shift_pct / 100.0))))
        base_revenue = float(row["price"]) * float(row["seats"]) * float(row["conv"])
        gross_revenue = scenario_price * scenario_seats * scenario_conv
        after_promo_revenue = gross_revenue * (1.0 - promo_discount_pct / 100.0)
        scenario_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "base_price": float(row["price"]),
                "base_conv": float(row["conv"]),
                "base_seats": float(row["seats"]),
                "base_revenue": base_revenue,
                "scenario_price": scenario_price,
                "scenario_conv": scenario_conv,
                "scenario_seats": scenario_seats,
                "scenario_revenue": gross_revenue,
                "scenario_revenue_after_promo": after_promo_revenue,
                "absolute_impact": after_promo_revenue - base_revenue,
            }
        )

    total_base_revenue = sum(r["base_revenue"] for r in scenario_rows)
    total_scenario_revenue = sum(r["scenario_revenue"] for r in scenario_rows)
    total_after_promo = sum(r["scenario_revenue_after_promo"] for r in scenario_rows)
    total_tickets = sum(r["scenario_seats"] * r["scenario_conv"] for r in scenario_rows)
    total_capacity = max(1.0, sum(r["scenario_seats"] for r in scenario_rows))
    scenario_conversion = total_tickets / total_capacity

    return {
        "rows": scenario_rows,
        "base_revenue": total_base_revenue,
        "scenario_revenue": total_scenario_revenue,
        "scenario_revenue_after_promo": total_after_promo,
        "scenario_tickets": total_tickets,
        "scenario_capacity": total_capacity,
        "scenario_conversion": scenario_conversion,
    }


def build_forecast_timeseries(
    scenario_metrics: Dict[str, Any],
    horizon_weeks: int,
    weekly_growth_pct: float,
    weekly_conversion_trend_pct: float,
) -> pd.DataFrame:
    horizon_weeks = max(1, int(horizon_weeks))
    weights = [idx for idx in range(1, horizon_weeks + 1)]
    weight_sum = float(sum(weights))

    base_weekly_revenue = float(scenario_metrics["scenario_revenue_after_promo"])
    base_weekly_tickets = float(scenario_metrics["scenario_tickets"])
    start_conversion = float(scenario_metrics["scenario_conversion"])

    cumulative_revenue = 0.0
    cumulative_tickets = 0.0
    rows: List[Dict[str, Any]] = []
    for week in range(1, horizon_weeks + 1):
        booking_weight = float(weights[week - 1]) / weight_sum
        growth_factor = (1.0 + weekly_growth_pct / 100.0) ** max(0, week - 1)
        conv_factor = (1.0 + weekly_conversion_trend_pct / 100.0) ** max(0, week - 1)
        week_conversion = clamp(start_conversion * conv_factor, 0.0, 1.0)
        week_revenue = base_weekly_revenue * booking_weight * growth_factor
        week_tickets = base_weekly_tickets * booking_weight * growth_factor
        cumulative_revenue += week_revenue
        cumulative_tickets += week_tickets
        rows.append(
            {
                "Week": week,
                "Weekly Revenue": week_revenue,
                "Cumulative Revenue": cumulative_revenue,
                "Weekly Tickets": week_tickets,
                "Cumulative Tickets": cumulative_tickets,
                "Conversion": week_conversion,
            }
        )

    return pd.DataFrame(rows)


def render_revenue_conversion_forecasting(pricing_result: Dict[str, Any], user_input: Dict[str, Any]) -> None:
    st.subheader("Revenue & Conversion Forecasting")
    if not pricing_result or not isinstance(pricing_result, dict):
        st.info("Run the pricing engine first to see revenue and conversion forecasting.")
        return
    st.write("Run what-if scenarios across price, demand, conversion, and promo effects with dynamic tier impact.")

    baseline = build_forecast_baseline(pricing_result, user_input)
    tier_rows = baseline.get("tier_rows", [])
    if not tier_rows:
        st.info("Pricing tiers are required to run forecasting.")
        return

    ranges = derive_dynamic_forecast_ranges(baseline)

    scenario_cols = st.columns(4)
    price_shift = scenario_cols[0].slider(
        "Price shift %",
        min_value=-float(ranges["price_shift_abs"]),
        max_value=float(ranges["price_shift_abs"]),
        value=0.0,
        step=0.5,
        key="forecast_price_shift",
    )
    conversion_shift = scenario_cols[1].slider(
        "Conversion shift %",
        min_value=-float(ranges["conv_shift_abs"]),
        max_value=float(ranges["conv_shift_abs"]),
        value=0.0,
        step=0.5,
        key="forecast_conversion_shift",
    )
    demand_shift = scenario_cols[2].slider(
        "Demand shift %",
        min_value=-float(ranges["demand_shift_abs"]),
        max_value=float(ranges["demand_shift_abs"]),
        value=0.0,
        step=0.5,
        key="forecast_demand_shift",
    )
    promo_discount = scenario_cols[3].slider(
        "Promo discount %",
        min_value=0.0,
        max_value=float(ranges["promo_max"]),
        value=0.0,
        step=0.5,
        key="forecast_promo_discount",
    )

    scenario_metrics = apply_pricing_scenario(
        tier_rows=tier_rows,
        price_shift_pct=price_shift,
        conversion_shift_pct=conversion_shift,
        demand_shift_pct=demand_shift,
        promo_discount_pct=promo_discount,
    )

    revenue_delta = scenario_metrics["scenario_revenue_after_promo"] - scenario_metrics["base_revenue"]
    conversion_delta = (scenario_metrics["scenario_conversion"] - baseline["base_conversion"]) * 100.0

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Base Revenue",
        f"₹{baseline['base_revenue']:,.0f}",
    )
    metric_cols[1].metric(
        "Scenario Revenue",
        f"₹{scenario_metrics['scenario_revenue_after_promo']:,.0f}",
        f"₹{revenue_delta:,.0f}",
    )
    metric_cols[2].metric(
        "Scenario Conversion",
        f"{scenario_metrics['scenario_conversion'] * 100:.2f}%",
        f"{conversion_delta:+.2f} pp",
    )
    metric_cols[3].metric(
        "Scenario Tickets",
        f"{scenario_metrics['scenario_tickets']:,.0f}",
    )

    # Tier impact analysis
    impact_rows = []
    for row in scenario_metrics["rows"]:
        base = float(row["base_revenue"])
        after = float(row["scenario_revenue_after_promo"])
        impact = after - base
        impact_rows.append(
            {
                "Tier": row["label"],
                "Base Revenue": base,
                "Scenario Revenue": after,
                "Impact": impact,
                "Impact %": (impact / base * 100.0) if base > 0 else 0.0,
                "Scenario Price": row["scenario_price"],
                "Scenario Conv %": row["scenario_conv"] * 100.0,
                "Scenario Seats": row["scenario_seats"],
            }
        )

    impact_df = pd.DataFrame(impact_rows)
    st.markdown("**Pricing Tier Impact Analysis**")
    impact_chart_df = impact_df.set_index("Tier")[["Base Revenue", "Scenario Revenue"]]
    st.bar_chart(impact_chart_df)
    st.dataframe(
        impact_df.style.format(
            {
                "Base Revenue": "₹{:,.0f}",
                "Scenario Revenue": "₹{:,.0f}",
                "Impact": "₹{:,.0f}",
                "Impact %": "{:+.2f}%",
                "Scenario Price": "₹{:,.0f}",
                "Scenario Conv %": "{:.2f}%",
                "Scenario Seats": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**What-if Revenue & Conversion Forecast**")
    horizon_cols = st.columns(2)
    horizon_weeks = horizon_cols[0].slider(
        "Forecast horizon (weeks)",
        min_value=2,
        max_value=26,
        value=12,
        step=1,
        key="forecast_horizon_weeks",
    )
    weekly_growth = horizon_cols[1].slider(
        "Weekly demand growth %",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        key="forecast_weekly_growth",
    )
    weekly_conversion_trend = st.slider(
        "Weekly conversion trend %",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.25,
        key="forecast_weekly_conversion_trend",
    )

    forecast_df = build_forecast_timeseries(
        scenario_metrics=scenario_metrics,
        horizon_weeks=horizon_weeks,
        weekly_growth_pct=weekly_growth,
        weekly_conversion_trend_pct=weekly_conversion_trend,
    )

    curve_df = forecast_df.set_index("Week")[["Weekly Revenue", "Cumulative Revenue", "Conversion"]]
    st.line_chart(curve_df)
    st.dataframe(
        forecast_df.style.format(
            {
                "Weekly Revenue": "₹{:,.0f}",
                "Cumulative Revenue": "₹{:,.0f}",
                "Weekly Tickets": "{:,.0f}",
                "Cumulative Tickets": "{:,.0f}",
                "Conversion": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_execution_monitor(result: Dict[str, Any]) -> None:
    st.subheader("Multi-Agent Execution Monitor")

    execution_logs = result.get("execution_logs", []) or []
    state_flow = result.get("state_flow", []) or []
    quality_results = result.get("quality_results", []) or []
    selected_agent = safe_text(result.get("selected_agent", "UNKNOWN"), "UNKNOWN")
    agent_sequence = result.get("agent_sequence", []) or []
    orchestration_plan = result.get("orchestration_plan", {}) or {}
    final_answer = safe_text(result.get("final_answer", result.get("sponsors_answer", "")), "")

    if not execution_logs and not state_flow and not quality_results and not agent_sequence:
        st.info("No execution trace available for this run yet.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Selected Agent", selected_agent)
    summary_cols[1].metric("Flow Steps", len(state_flow))
    summary_cols[2].metric("Log Events", len(execution_logs))
    summary_cols[3].metric("Quality Checks", len(quality_results))

    if agent_sequence or orchestration_plan:
        st.markdown("**Orchestration Plan**")
        plan_cols = st.columns(2)
        if agent_sequence:
            plan_cols[0].write(f"**Agent Sequence**: {', '.join(agent_sequence)}")
        if orchestration_plan:
            plan_cols[1].json(orchestration_plan)

    if final_answer:
        st.markdown("**Final Combined Answer**")
        st.markdown(final_answer)

    st.markdown("**State Flow Visualization**")
    if state_flow:
        st.code(" -> ".join(state_flow), language="text")
        flow_frame = pd.DataFrame(
            [{"Step": idx + 1, "Node": node} for idx, node in enumerate(state_flow)]
        )
        st.dataframe(flow_frame, use_container_width=True, hide_index=True)
    else:
        st.caption("No state flow recorded.")

    st.markdown("**Agent Execution Logs / Tracking**")
    if execution_logs:
        st.dataframe(pd.DataFrame(execution_logs), use_container_width=True, hide_index=True)
    else:
        st.caption("No execution logs recorded.")

    st.markdown("**Quality Control Results**")
    qc_cols = st.columns(2)
    qc_cols[0].metric("Hallucination Verdict", safe_text(result.get("hallucination_verdict", "N/A"), "N/A"))
    qc_cols[1].metric("Usefulness Verdict", safe_text(result.get("usefulness_verdict", "N/A"), "N/A"))
    if quality_results:
        st.dataframe(pd.DataFrame(quality_results), use_container_width=True, hide_index=True)
    else:
        st.caption("No quality-control checks recorded.")


def render_internal_processing_visualization(planner_result: Dict[str, Any], pricing_result: Dict[str, Any]) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(" Internal Processing Visualization")
    st.write("Live terminal-style view of internal processing prints and execution logs.")
    st.markdown("</div>", unsafe_allow_html=True)

    if "terminal_output_lines" not in st.session_state:
        st.session_state["terminal_output_lines"] = []

    terminal_lines = list(st.session_state.get("terminal_output_lines", []))
    if not terminal_lines:
        fallback_logs: List[str] = []
        for source in [planner_result or {}, pricing_result or {}]:
            fallback_logs.extend(source.get("logs", []) or [])
        terminal_lines = fallback_logs

    controls = st.columns(3)
    max_visible = controls[0].slider(
        "Visible lines",
        min_value=50,
        max_value=2000,
        value=400,
        step=50,
        key="terminal_visible_lines",
    )
    controls[1].metric("Buffered lines", len(st.session_state.get("terminal_output_lines", [])))
    if controls[2].button("Clear Terminal", key="clear_internal_terminal", use_container_width=True):
        st.session_state["terminal_output_lines"] = []
        st.rerun()

    visible_lines = terminal_lines[-max_visible:] if terminal_lines else []
    if not visible_lines:
        st.info("No internal output captured yet. Run planner/pricing/email actions to populate this view.")
        return

    terminal_text = "\n".join(visible_lines)
    st.markdown(
        f"""
        <div style="
            background:#0b0f14;
            color:#d6deeb;
            border:1px solid #1f2a37;
            border-radius:12px;
            padding:14px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
        ">
            <div style="font-size:0.8rem;color:#8ba1b3;margin-bottom:8px;">terminal://agentic-event-planner</div>
            <pre style="margin:0;white-space:pre-wrap;word-break:break-word;font-family:Consolas,'Courier New',monospace;font-size:0.84rem;line-height:1.35;">{html.escape(terminal_text)}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_langsmith_client() -> Any:
    try:
        from langsmith import Client
    except Exception as exc:
        raise RuntimeError(f"LangSmith SDK unavailable: {exc}")

    api_key = safe_text(get_runtime_setting("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY) or get_runtime_setting("LANGSMITH_API_KEY", LANGSMITH_API_KEY))
    if not api_key:
        raise RuntimeError("Missing LangSmith API key. Add it in Settings.")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    endpoint = safe_text(get_runtime_setting("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT))
    if endpoint:
        kwargs["api_url"] = endpoint
    return Client(**kwargs)


def _safe_iso_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return safe_text(value)


def _latency_to_ms(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds()) * 1000.0
    try:
        return float(value) * 1000.0 if float(value) < 1000 else float(value)
    except Exception:
        return 0.0


def _run_to_json_text(run: Any) -> str:
    try:
        if hasattr(run, "model_dump"):
            payload = run.model_dump()
        elif hasattr(run, "dict"):
            payload = run.dict()
        else:
            payload = dict(getattr(run, "__dict__", {}))
        return json.dumps(payload, ensure_ascii=True, default=str)
    except Exception:
        return json.dumps({"run": str(run)}, ensure_ascii=True)


def fetch_langsmith_traces(project_name: str, limit: int = 100) -> pd.DataFrame:
    client = get_langsmith_client()
    runs = list(client.list_runs(project_name=project_name, limit=max(1, int(limit))))

    rows: List[Dict[str, Any]] = []
    for run in runs:
        run_id = safe_text(getattr(run, "id", ""))
        status = safe_text(getattr(run, "status", ""), "unknown")
        error_text = safe_text(getattr(run, "error", ""))
        if error_text and status == "unknown":
            status = "error"
        rows.append(
            {
                "run_id": run_id,
                "name": safe_text(getattr(run, "name", "")),
                "run_type": safe_text(getattr(run, "run_type", "")),
                "status": status,
                "error": error_text,
                "start_time": _safe_iso_text(getattr(run, "start_time", None)),
                "end_time": _safe_iso_text(getattr(run, "end_time", None)),
                "latency_ms": round(_latency_to_ms(getattr(run, "latency", 0.0)), 2),
                "total_tokens": int(getattr(run, "total_tokens", 0) or 0),
                "prompt_tokens": int(getattr(run, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(run, "completion_tokens", 0) or 0),
                "project_name": project_name,
                "raw_json": _run_to_json_text(run),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "name",
                "run_type",
                "status",
                "error",
                "start_time",
                "end_time",
                "latency_ms",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "project_name",
                "raw_json",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["start_ts"] = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)
    frame = frame.sort_values(by="start_ts", ascending=False, na_position="last").reset_index(drop=True)
    return frame


def render_langsmith_tracing_tab() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("LangSmith Tracing")
    st.write("Trace monitoring, run health alerts, and detailed LangSmith outputs.")
    st.markdown("</div>", unsafe_allow_html=True)

    controls = st.columns(5)
    project_name = controls[0].text_input("Project", value=get_runtime_setting("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT), key="ls_project_main")
    limit = controls[1].slider("Rows", min_value=20, max_value=1000, value=200, step=20, key="ls_limit_main")
    failed_only = controls[2].checkbox("Failed only", value=False, key="ls_failed_only")
    stale_minutes = controls[3].slider("Stale threshold (min)", min_value=1, max_value=120, value=15, step=1, key="ls_stale_mins")
    latency_alert_ms = controls[4].slider("Latency alert (ms)", min_value=100, max_value=10000, value=2000, step=100, key="ls_latency_alert")

    refresh = st.button("Refresh Traces", key="ls_refresh_main", use_container_width=True)
    if refresh or "ls_traces_df_main" not in st.session_state:
        try:
            st.session_state["ls_traces_df_main"] = fetch_langsmith_traces(project_name=project_name, limit=limit)
            st.session_state["ls_error_main"] = ""
        except Exception as exc:
            st.session_state["ls_error_main"] = str(exc)
            st.session_state["ls_traces_df_main"] = pd.DataFrame()

    err = safe_text(st.session_state.get("ls_error_main", ""))
    if err:
        st.error(f"LangSmith trace fetch failed: {err}")
        return

    traces_df = st.session_state.get("ls_traces_df_main", pd.DataFrame()).copy()
    if traces_df.empty:
        st.info("No LangSmith traces found for this project.")
        return

    if failed_only:
        traces_df = traces_df[(traces_df["status"].str.lower().isin(["error", "failed"])) | (traces_df["error"].astype(str).str.len() > 0)]
        if traces_df.empty:
            st.info("No failed traces for current filters.")
            return

    total_runs = int(len(traces_df))
    failed_runs = int(((traces_df["status"].str.lower().isin(["error", "failed"])) | (traces_df["error"].astype(str).str.len() > 0)).sum())
    error_rate = (failed_runs / max(total_runs, 1)) * 100.0
    avg_latency_ms = float(pd.to_numeric(traces_df["latency_ms"], errors="coerce").fillna(0).mean())
    total_tokens = int(pd.to_numeric(traces_df["total_tokens"], errors="coerce").fillna(0).sum())

    latest_ts = pd.to_datetime(traces_df["start_time"], errors="coerce", utc=True).max()
    if pd.isna(latest_ts):
        last_trace_age_min = float("inf")
    else:
        last_trace_age_min = (datetime.now(timezone.utc) - latest_ts.to_pydatetime()).total_seconds() / 60.0

    summary_cols = st.columns(6)
    summary_cols[0].metric("Runs", f"{total_runs:,}")
    summary_cols[1].metric("Failed", f"{failed_runs:,}")
    summary_cols[2].metric("Error Rate", f"{error_rate:.2f}%")
    summary_cols[3].metric("Avg Latency", f"{avg_latency_ms:.1f} ms")
    summary_cols[4].metric("Total Tokens", f"{total_tokens:,}")
    summary_cols[5].metric("Last Trace Age", "N/A" if last_trace_age_min == float("inf") else f"{last_trace_age_min:.1f} min")

    if failed_runs > 0:
        st.warning(f"Alert: {failed_runs} trace(s) failed or contain errors.")
    if avg_latency_ms > float(latency_alert_ms):
        st.warning(f"Alert: Average latency {avg_latency_ms:.1f} ms exceeded threshold {latency_alert_ms} ms.")
    if last_trace_age_min != float("inf") and last_trace_age_min > float(stale_minutes):
        st.warning(f"Alert: Last trace is stale ({last_trace_age_min:.1f} min old).")

    status_counts = traces_df.groupby("status", dropna=False).size().reset_index(name="count")
    type_counts = traces_df.groupby("run_type", dropna=False).size().reset_index(name="count")

    viz_cols = st.columns(2)
    with viz_cols[0]:
        st.markdown("**Status Distribution**")
        st.bar_chart(status_counts.set_index("status"))
    with viz_cols[1]:
        st.markdown("**Run Type Distribution**")
        st.bar_chart(type_counts.set_index("run_type"))

    traces_ts = traces_df.copy()
    traces_ts["start_ts"] = pd.to_datetime(traces_ts["start_time"], errors="coerce", utc=True)
    traces_ts = traces_ts.dropna(subset=["start_ts"]).sort_values("start_ts")
    if not traces_ts.empty:
        trend_df = traces_ts.set_index("start_ts")[["latency_ms", "total_tokens"]]
        st.markdown("**Latency and Token Trend**")
        st.line_chart(trend_df)

    st.markdown("**Trace Table**")
    table_cols = [
        "run_id",
        "name",
        "run_type",
        "status",
        "error",
        "start_time",
        "end_time",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    st.dataframe(traces_df[table_cols], use_container_width=True, hide_index=True)

    st.markdown("**Trace Detail JSON**")
    run_ids = traces_df["run_id"].astype(str).tolist()
    selected_run = st.selectbox("Select run id", options=run_ids, key="ls_selected_run_main")
    selected_row = traces_df[traces_df["run_id"].astype(str) == str(selected_run)].head(1)
    if not selected_row.empty:
        raw_text = safe_text(selected_row.iloc[0].get("raw_json", "{}"), "{}")
        try:
            st.json(json.loads(raw_text))
        except Exception:
            st.code(raw_text, language="json")


def _get_or_create_streamlit_session_id() -> str:
    if "app_session_id" not in st.session_state:
        st.session_state["app_session_id"] = uuid.uuid4().hex
    return safe_text(st.session_state.get("app_session_id"), uuid.uuid4().hex)


def _get_session_history_db_path() -> str:
    session_id = _get_or_create_streamlit_session_id()
    os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
    return os.path.join(SESSION_HISTORY_DIR, f"history_{session_id}.sqlite3")


def _ensure_session_history_table() -> str:
    db_path = _get_session_history_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                query_text TEXT NOT NULL,
                final_output TEXT NOT NULL,
                selected_agent TEXT,
                route_target TEXT,
                source TEXT
            )
            """
        )
        conn.commit()
    return db_path


def log_session_history_entry(
    query_text: str,
    final_output: str,
    selected_agent: str,
    route_target: str,
    source: str,
) -> None:
    q_text = safe_text(query_text)
    output_text = safe_text(final_output)
    if not q_text and not output_text:
        return
    db_path = _ensure_session_history_table()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO history (created_at, query_text, final_output, selected_agent, route_target, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                q_text,
                output_text,
                safe_text(selected_agent),
                safe_text(route_target),
                safe_text(source),
            ),
        )
        conn.commit()


def read_session_history_entries(limit: int = 200) -> pd.DataFrame:
    db_path = _ensure_session_history_table()
    query = (
        "SELECT id, created_at, source, selected_agent, route_target, query_text, final_output "
        "FROM history ORDER BY id DESC LIMIT ?"
    )
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(query, conn, params=[max(1, int(limit))])
    return frame


def render_history_tab() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(" History")
    st.write("Per-session history database showing user queries and final agent outputs.")
    st.markdown("</div>", unsafe_allow_html=True)

    session_id = _get_or_create_streamlit_session_id()
    db_path = _ensure_session_history_table()

    meta_cols = st.columns(2)
    meta_cols[0].write(f"**Session ID**: {session_id}")
    meta_cols[1].write(f"**DB Path**: {db_path}")

    max_rows = st.slider("Rows to show", min_value=20, max_value=1000, value=200, step=20, key="history_max_rows")
    history_df = read_session_history_entries(limit=max_rows)

    if history_df.empty:
        st.info("No history entries yet for this session.")
        return

    st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download History CSV",
        data=history_df.to_csv(index=False),
        file_name=f"session_history_{session_id}.csv",
        mime="text/csv",
        key="download_session_history",
    )


def render_status_badges() -> None:
    cols = st.columns(4)
    cols[0].metric("OpenRouter", "Ready" if get_runtime_setting("OPENROUTER_API_KEY", OPENROUTER_API_KEY) else "Missing")
    cols[1].metric("Tavily", "Ready" if get_runtime_setting("TAVILY_API_KEY", TAVILY_API_KEY) else "Missing")
    cols[2].metric("LangSmith", "Ready" if (get_runtime_setting("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY) or get_runtime_setting("LANGSMITH_API_KEY", LANGSMITH_API_KEY)) else "Missing")
    cols[3].metric("Vector DBs", "Loaded")


def render_documents(docs: List[Document]) -> None:
    if not docs:
        st.info("No matching documents found.")
        return
    frame = []
    for doc in docs[:8]:
        frame.append({
            "name": doc_name(doc),
            "metadata": json.dumps(doc.metadata or {}, ensure_ascii=True),
            "content": doc.page_content[:280] + ("..." if len(doc.page_content) > 280 else ""),
        })
    st.dataframe(pd.DataFrame(frame), use_container_width=True)


def contacts_from_docs(docs: List[Document]) -> List[Dict[str, str]]:
    contacts: List[Dict[str, str]] = []
    seen_emails = set()
    for doc in docs:
        md = dict(doc.metadata or {})
        email = safe_text(md.get("email"))
        if not email:
            match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", safe_text(doc.page_content))
            email = safe_text(match.group(0)) if match else ""
        if not email or email in seen_emails:
            continue
        seen_emails.add(email)
        contacts.append(
            {
                "name": safe_text(md.get("name"), email.split("@")[0]),
                "email": email,
                "company": safe_text(md.get("company"), "Unknown"),
                "role": safe_text(md.get("role"), "Unknown"),
                "type": safe_text(md.get("contact_type") or md.get("type"), "unknown"),
                "industry": safe_text(md.get("industry"), "unknown"),
            }
        )
    return contacts


def load_contacts_for_outreach(query: str, top_k: int = 20) -> List[Dict[str, str]]:
    db = get_contact_vectordb()
    if db is not None:
        try:
            docs = db.similarity_search(query, k=max(1, top_k))
            contacts = contacts_from_docs(docs)
            if contacts:
                return contacts
        except Exception:
            pass
    if os.path.exists(CONTACTS_CSV_PATH):
        try:
            contacts_df = pd.read_csv(CONTACTS_CSV_PATH).head(max(1, top_k))
            contacts: List[Dict[str, str]] = []
            for _, row in contacts_df.iterrows():
                email = safe_text(row.get("email"))
                if not email:
                    continue
                contacts.append(
                    {
                        "name": safe_text(row.get("name"), email.split("@")[0]),
                        "email": email,
                        "company": safe_text(row.get("company"), "Unknown"),
                        "role": safe_text(row.get("role"), "Unknown"),
                        "type": safe_text(row.get("type"), "unknown"),
                        "industry": safe_text(row.get("industry"), "unknown"),
                    }
                )
            return contacts
        except Exception:
            return []
    return []


def generate_contact_draft(
    user_input: Dict[str, Any],
    contact: Dict[str, str],
    campaign_brief: str,
    sender_name: str,
    tone: str,
) -> Dict[str, str]:
    prompt = (
        "You generate personalized outreach emails for event planning.\\n"
        "Return strict JSON only with keys: subject, body.\\n\\n"
        f"Event details: {json.dumps(user_input, ensure_ascii=True)}\\n"
        f"Contact: {json.dumps(contact, ensure_ascii=True)}\\n"
        f"Campaign brief: {campaign_brief}\\n"
        f"Tone: {tone}\\n"
        f"Sender name: {sender_name}\\n\\n"
        "Rules:\\n"
        "- Keep subject concise and relevant.\\n"
        "- Keep body practical and personalized to the contact role/company.\\n"
        "- Include a clear CTA and no fake claims."
    )
    raw = call_llm(prompt)
    parsed = _parse_jsonish(raw)
    subject = safe_text(parsed.get("subject"), f"Opportunity for {contact.get('name', 'your team')}")
    body = safe_text(parsed.get("body"))
    if not body:
        body = (
            f"Hi {contact.get('name', 'there')},\\n\\n"
            f"{campaign_brief}\\n\\n"
            "Would you be open to a short conversation this week?\\n\\n"
            f"Best regards,\\n{sender_name}" if sender_name else f"Hi {contact.get('name', 'there')},\\n\\n{campaign_brief}"
        )
    return {"subject": subject, "body": body}


def send_email_via_smtp(
    sender_email: str,
    app_password: str,
    receiver_email: str,
    subject: str,
    message_text: str,
) -> Dict[str, str]:
    try:
        msg = MIMEText(message_text)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)

        return {"status": "sent", "message": "Email sent successfully"}
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}


def render_email_outreach_tab(user_input: Dict[str, Any], planner_result: Dict[str, Any]) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Email Outreach")
    st.write("Generate personalized drafts, review by recipient, approve some or all, then send via SMTP.")
    st.markdown("</div>", unsafe_allow_html=True)

    if "email_contacts" not in st.session_state:
        st.session_state["email_contacts"] = []
    if "email_drafts_df" not in st.session_state:
        st.session_state["email_drafts_df"] = pd.DataFrame()

    default_query = safe_text(user_input.get("query"), "event outreach contacts")
    contact_query = st.text_input("Contact retrieval query", value=default_query, key="email_contact_query")
    top_k = st.slider("Contacts to retrieve", min_value=1, max_value=50, value=15, step=1, key="email_top_k")

    if st.button("Load Contacts", key="load_outreach_contacts", use_container_width=True):
        contacts = load_contacts_for_outreach(contact_query, top_k=top_k)
        st.session_state["email_contacts"] = contacts
        if contacts:
            st.success(f"Loaded {len(contacts)} contacts.")
        else:
            st.warning("No contacts found. Upload contacts data first.")

    contacts = st.session_state.get("email_contacts", [])
    if contacts:
        st.markdown("**Retrieved Contacts**")
        st.dataframe(pd.DataFrame(contacts), use_container_width=True, hide_index=True)
    else:
        st.info("Load contacts to start generating drafts.")
        return

    st.divider()
    st.subheader("Generate Drafts")
    campaign_brief_default = safe_text(planner_result.get("final_answer") or planner_result.get("sponsors_answer"), "")
    campaign_brief = st.text_area(
        "Campaign brief",
        value=campaign_brief_default,
        height=180,
        placeholder="Describe what these emails should communicate.",
        key="email_campaign_brief",
    )
    sender_name = st.text_input("Sender display name", value="", key="email_sender_name")
    tone = st.selectbox("Tone", options=["Professional", "Friendly", "Concise", "Persuasive"], key="email_tone")

    if st.button("Generate Drafts", key="generate_email_drafts", use_container_width=True):
        draft_rows: List[Dict[str, Any]] = []
        with st.spinner("Generating personalized drafts..."):
            for contact in contacts:
                draft = generate_contact_draft(
                    user_input=user_input,
                    contact=contact,
                    campaign_brief=campaign_brief,
                    sender_name=sender_name,
                    tone=tone,
                )
                draft_rows.append(
                    {
                        "approve": True,
                        "name": contact.get("name", ""),
                        "email": contact.get("email", ""),
                        "subject": draft.get("subject", ""),
                        "body": draft.get("body", ""),
                        "status": "drafted",
                        "send_message": "",
                    }
                )
        st.session_state["email_drafts_df"] = pd.DataFrame(draft_rows)
        st.success(f"Generated {len(draft_rows)} draft emails.")

    drafts_df = st.session_state.get("email_drafts_df", pd.DataFrame())
    if drafts_df.empty:
        st.info("Generate drafts to review and approve emails.")
        return

    st.divider()
    st.subheader("Approve and Send")
    edited_df = st.data_editor(
        drafts_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "approve": st.column_config.CheckboxColumn("Approve", default=True),
            "email": st.column_config.TextColumn("Email", disabled=True),
            "name": st.column_config.TextColumn("Name", disabled=True),
            "subject": st.column_config.TextColumn("Subject"),
            "body": st.column_config.TextColumn("Body", width="large"),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "send_message": st.column_config.TextColumn("Send Message", disabled=True, width="large"),
        },
        key="email_drafts_editor",
    )
    st.session_state["email_drafts_df"] = edited_df

    st.markdown("**SMTP Credentials**")
    sender_email = st.text_input("Sender Gmail", value="", key="smtp_sender_email", placeholder="sender@gmail.com")
    app_password = st.text_input(
        "Google App Password",
        value="",
        type="password",
        key="smtp_app_password",
        placeholder="Use a Gmail app password",
    )

    send_cols = st.columns(2)
    send_approved = send_cols[0].button("Send Approved", key="send_approved_email", use_container_width=True)
    send_all = send_cols[1].button("Send All Drafts", key="send_all_email", use_container_width=True)

    if send_approved or send_all:
        if not sender_email.strip() or not app_password.strip():
            st.error("Enter sender Gmail and Google app password to send emails.")
            return

        send_df = st.session_state.get("email_drafts_df", pd.DataFrame()).copy()
        if send_df.empty:
            st.warning("No drafts available to send.")
            return

        if send_all:
            target_idx = send_df.index.tolist()
        else:
            target_idx = send_df.index[send_df["approve"] == True].tolist()

        if not target_idx:
            st.warning("No approved drafts selected.")
            return

        with st.spinner("Sending emails via SMTP..."):
            sent_count = 0
            fail_count = 0
            for idx in target_idx:
                receiver_email = safe_text(send_df.at[idx, "email"])
                subject = safe_text(send_df.at[idx, "subject"])
                body = safe_text(send_df.at[idx, "body"])
                if not receiver_email:
                    send_df.at[idx, "status"] = "failed"
                    send_df.at[idx, "send_message"] = "Missing receiver email"
                    fail_count += 1
                    continue
                result = send_email_via_smtp(
                    sender_email=sender_email.strip(),
                    app_password=app_password.strip(),
                    receiver_email=receiver_email,
                    subject=subject,
                    message_text=body,
                )
                send_df.at[idx, "status"] = result.get("status", "failed")
                send_df.at[idx, "send_message"] = result.get("message", "")
                if result.get("status") == "sent":
                    sent_count += 1
                else:
                    fail_count += 1

        st.session_state["email_drafts_df"] = send_df
        if fail_count == 0:
            st.success(f"Sent {sent_count} emails successfully.")
        else:
            st.warning(f"Sent {sent_count} emails, {fail_count} failed. Check the status column for details.")


# ═══════════════════════════════════════════════════════════════════════════
# CSV UPLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def read_csv_upload(uploaded_file: Any) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception:
        return None


def save_uploaded_csv(uploaded_file: Any, target_path: str) -> bool:
    if uploaded_file is None:
        return False
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as handle:
            handle.write(uploaded_file.getvalue())
        return True
    except Exception:
        return False


def build_documents_from_uploads(dataframes: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, List[Document]]:
    events_df = dataframes.get("events")
    sponsors_df = dataframes.get("sponsors")
    speakers_df = dataframes.get("speakers")
    exhibitors_df = dataframes.get("exhibitors")
    venues_df = dataframes.get("venues")
    communities_df = dataframes.get("communities")
    sessions_df = dataframes.get("sessions")
    rooms_df = dataframes.get("rooms")
    time_slots_df = dataframes.get("time_slots")
    pricing_df = dataframes.get("pricing_tiers")
    contacts_df = dataframes.get("contacts")
    documents: Dict[str, List[Document]] = {
        "sponsor": [],
        "speaker": [],
        "exhibitor": [],
        "venue": [],
        "community": [],
        "event_ops": [],
        "pricing": [],
        "contact": [],
    }
    if sponsors_df is not None:
        for _, row in sponsors_df.iterrows():
            text = (
                f"{safe_text(row.get('sponsor_name'), 'Unknown Sponsor')} is a {safe_text(row.get('industry'), 'company')} company "
                f"that sponsored {safe_text(row.get('event_name'), 'Unknown Event')}, "
                f"a {safe_text(row.get('category'), 'conference')} conference held in "
                f"{safe_text(row.get('city'), 'Unknown')}, {safe_text(row.get('country'), 'Unknown')} "
                f"in {safe_text(row.get('year'), 'Unknown')}, as a {safe_text(row.get('tier'), 'unknown')} tier sponsor."
            )
            documents["sponsor"].append(Document(page_content=text, metadata={"type": "sponsor", "name": safe_text(row.get("sponsor_name"))}))

    if speakers_df is not None:
        for _, row in speakers_df.iterrows():
            text = (
                f"{safe_text(row.get('speaker_name'), 'Unknown Speaker')} is a {safe_text(row.get('title'), 'speaker')} "
                f"from {safe_text(row.get('company'), 'Unknown Company')} with expertise in {safe_text(row.get('expertise'), 'general topics')}. "
                f"Spoke at {safe_text(row.get('event_name'), 'Unknown Event')} in {safe_text(row.get('city'), 'Unknown City')}."
            )
            documents["speaker"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "speaker",
                        "speaker_name": safe_text(row.get("speaker_name")),
                        "company": safe_text(row.get("company")),
                        "expertise": safe_text(row.get("expertise")),
                    },
                )
            )

    if exhibitors_df is not None:
        for _, row in exhibitors_df.iterrows():
            text = (
                f"{safe_text(row.get('company_name'), 'Unknown Company')} exhibited at "
                f"{safe_text(row.get('event_name'), 'Unknown Event')} in {safe_text(row.get('city'), 'Unknown City')}. "
                f"Industry: {safe_text(row.get('industry'), 'unknown')}. Booth type: {safe_text(row.get('booth_type'), 'standard')}."
            )
            documents["exhibitor"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "exhibitor",
                        "company_name": safe_text(row.get("company_name")),
                        "industry": safe_text(row.get("industry")),
                        "booth_type": safe_text(row.get("booth_type")),
                    },
                )
            )

    if venues_df is not None:
        for _, row in venues_df.iterrows():
            text = (
                f"{safe_text(row.get('venue_name'), 'Unknown Venue')} is located in "
                f"{safe_text(row.get('city'), 'Unknown City')}, {safe_text(row.get('country'), 'Unknown Country')} with capacity "
                f"{safe_text(row.get('capacity'), 'unknown')}. Venue type: {safe_text(row.get('venue_type'), 'unknown')}."
            )
            documents["venue"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "venue",
                        "venue_name": safe_text(row.get("venue_name")),
                        "city": safe_text(row.get("city")),
                        "country": safe_text(row.get("country")),
                        "capacity": safe_text(row.get("capacity")),
                    },
                )
            )

    if communities_df is not None:
        for _, row in communities_df.iterrows():
            text = (
                f"{safe_text(row.get('community_name'), 'Unknown Community')} is a {safe_text(row.get('platform'), 'community')} "
                f"focused on {safe_text(row.get('topic'), 'technology')} with approximately {safe_text(row.get('members'), 'unknown')} members. "
                f"Primary region: {safe_text(row.get('region'), 'global')}."
            )
            documents["community"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "community",
                        "community_name": safe_text(row.get("community_name")),
                        "platform": safe_text(row.get("platform")),
                        "topic": safe_text(row.get("topic")),
                    },
                )
            )

    if sessions_df is not None:
        for _, row in sessions_df.iterrows():
            text = (
                f"Session {safe_text(row.get('session_id'), 'unknown')} titled {safe_text(row.get('session_title'), 'Untitled')} "
                f"track {safe_text(row.get('track'), 'general')} duration {safe_text(row.get('duration_min'), 'unknown')} minutes "
                f"speaker {safe_text(row.get('speaker_name'), 'TBD')}."
            )
            documents["event_ops"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "session",
                        "session_id": safe_text(row.get("session_id")),
                        "track": safe_text(row.get("track")),
                        "speaker_name": safe_text(row.get("speaker_name")),
                    },
                )
            )

    if rooms_df is not None:
        for _, row in rooms_df.iterrows():
            text = (
                f"Room {safe_text(row.get('room_name'), 'Unknown Room')} capacity {safe_text(row.get('capacity'), 'unknown')} "
                f"location zone {safe_text(row.get('zone'), 'general')} with setup {safe_text(row.get('setup_type'), 'standard')}."
            )
            documents["event_ops"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "room",
                        "room_name": safe_text(row.get("room_name")),
                        "capacity": safe_text(row.get("capacity")),
                    },
                )
            )

    if time_slots_df is not None:
        for _, row in time_slots_df.iterrows():
            text = (
                f"Time slot {safe_text(row.get('slot_id'), 'unknown')} from {safe_text(row.get('start_time'), 'unknown')} "
                f"to {safe_text(row.get('end_time'), 'unknown')} day {safe_text(row.get('day'), 'unknown')} "
                f"type {safe_text(row.get('slot_type'), 'session')}."
            )
            documents["event_ops"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "time_slot",
                        "slot_id": safe_text(row.get("slot_id")),
                        "day": safe_text(row.get("day")),
                        "slot_type": safe_text(row.get("slot_type")),
                    },
                )
            )

    if pricing_df is not None:
        for _, row in pricing_df.iterrows():
            text = (
                f"Pricing tier {safe_text(row.get('tier_name'), 'Unknown Tier')} for event category "
                f"{safe_text(row.get('category'), 'conference')} has base price {safe_text(row.get('base_price'), 'unknown')} "
                f"expected conversion {safe_text(row.get('expected_conversion'), 'unknown')} and allocation {safe_text(row.get('seat_allocation'), 'unknown')} seats."
            )
            documents["pricing"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "pricing_tier",
                        "tier_name": safe_text(row.get("tier_name")),
                        "category": safe_text(row.get("category")),
                        "base_price": safe_text(row.get("base_price")),
                    },
                )
            )

    if events_df is not None:
        for _, row in events_df.iterrows():
            pricing_text = (
                f"Historical event {safe_text(row.get('event_name'), 'Unknown Event')} in "
                f"{safe_text(row.get('city'), 'Unknown City')} category {safe_text(row.get('category'), 'conference')} "
                f"attendance {safe_text(row.get('attendance'), 'unknown')} had ticket baseline around "
                f"{safe_text(row.get('avg_ticket_price'), safe_text(row.get('ticket_price'), 'unknown'))}."
            )
            documents["pricing"].append(
                Document(
                    page_content=pricing_text,
                    metadata={
                        "type": "event_history",
                        "event_name": safe_text(row.get("event_name")),
                        "city": safe_text(row.get("city")),
                        "category": safe_text(row.get("category")),
                    },
                )
            )

    if contacts_df is not None:
        for _, row in contacts_df.iterrows():
            text = (
                f"{safe_text(row.get('name'), 'Unknown Contact')} works at {safe_text(row.get('company'), 'Unknown Company')} "
                f"as {safe_text(row.get('role'), 'Unknown Role')}. Type: {safe_text(row.get('type'), 'unknown')}. "
                f"Industry: {safe_text(row.get('industry'), 'unknown')}. Email: {safe_text(row.get('email'), 'missing')}."
            )
            documents["contact"].append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "contact",
                        "name": safe_text(row.get("name")),
                        "company": safe_text(row.get("company")),
                        "role": safe_text(row.get("role")),
                        "email": safe_text(row.get("email")),
                    },
                )
            )

    return documents


def ingest_uploaded_dataframes(dataframes: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, int]:
    if has_ingestion_been_done():
        return {}
    documents_by_collection = build_documents_from_uploads(dataframes)
    counts: Dict[str, int] = {}
    for collection_name, docs in documents_by_collection.items():
        if not docs:
            continue
        target_dir = VECTOR_DB_PATHS[collection_name]
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        db = Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            persist_directory=target_dir,
        )
        db.persist()
        counts[collection_name] = len(docs)
    clear_resource_caches()
    return counts


def load_ingestion_status_from_disk() -> Dict[str, Any]:
    if not os.path.exists(INGESTION_STATUS_PATH):
        return {}
    try:
        with open(INGESTION_STATUS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_ingestion_status_to_disk(status: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(INGESTION_STATUS_PATH), exist_ok=True)
    with open(INGESTION_STATUS_PATH, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=True, indent=2)


def has_ingestion_been_done() -> bool:
    status = load_ingestion_status_from_disk()
    return bool(status.get("complete", False))


def render_dataframe_preview(title: str, dataframe: Optional[pd.DataFrame], source_path: str) -> None:
    st.markdown(f"**{title}**")
    if dataframe is not None and not dataframe.empty:
        st.caption("📤 Uploaded preview")
        st.dataframe(dataframe.head(5), use_container_width=True, hide_index=True)
        return
    if os.path.exists(source_path):
        try:
            existing = pd.read_csv(source_path)
            st.caption(f"File on disk: {source_path}")
            st.dataframe(existing.head(5), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not read {source_path}: {exc}")
    else:
        st.info("No file uploaded yet.")


def initialize_ingestion_state() -> None:
    """Initialize ingestion tracking in session state."""
    disk_status = load_ingestion_status_from_disk()
    if "ingestion_history" not in st.session_state:
        st.session_state["ingestion_history"] = disk_status.get("history", {})
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {}
    if "ingestion_complete" not in st.session_state:
        st.session_state["ingestion_complete"] = bool(disk_status.get("complete", False))


def get_ingestion_status() -> Dict[str, Any]:
    """Get the current ingestion status."""
    return {
        "history": st.session_state.get("ingestion_history", {}),
        "complete": st.session_state.get("ingestion_complete", False),
        "uploaded_files": st.session_state.get("uploaded_files", {}),
    }


def mark_ingestion_complete(counts: Dict[str, int]) -> None:
    """Mark ingestion as complete and store counts."""
    history = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "status": "success",
    }
    st.session_state["ingestion_complete"] = True
    st.session_state["ingestion_history"] = history
    save_ingestion_status_to_disk({"complete": True, "history": history})


def render_ingestion_status() -> None:
    """Display ingestion status."""
    status = get_ingestion_status()
    if status["complete"]:
        history = status.get("history", {})
        if history:
            st.success("Data ingestion completed")
            with st.expander("Ingestion details", expanded=False):
                st.write(f"**Timestamp**: {history.get('timestamp')}")
                counts = history.get("counts", {})
                if counts:
                    st.write("**Collections created/updated**:")
                    for collection, count in counts.items():
                        st.write(f"- {collection}: {count} documents")
    else:
        st.info("No ingestion history yet. Upload and process CSV files to ingest data.")


def render_csv_upload_interface() -> None:
    """Render full CSV upload and ingestion interface."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data Upload and Ingestion")
    st.markdown("Upload CSV files to build/rebuild vector databases. **First ingestion will process all files.**")
    st.markdown("</div>", unsafe_allow_html=True)
    
    initialize_ingestion_state()
    
    # Show current ingestion status
    render_ingestion_status()
    
    st.divider()
    
    ingestion_locked = has_ingestion_been_done()

    # File upload section
    st.subheader("Upload Datasets")
    
    uploaded_files: Dict[str, Any] = {}
    cols = st.columns(2)
    
    for idx, (key, spec) in enumerate(DATASET_SPECS.items()):
        col = cols[idx % 2]
        with col:
            label = spec["label"]
            uploaded = col.file_uploader(
                f"{label}",
                type=["csv"],
                key=f"upload_{key}",
                help=f"Upload {label} CSV file",
            )
            if uploaded:
                uploaded_files[key] = uploaded
                st.session_state["uploaded_files"][key] = uploaded.name
    
    st.divider()
    
    # Preview uploaded files
    if uploaded_files:
        st.subheader("Preview Uploaded Files")
        dataframes: Dict[str, Optional[pd.DataFrame]] = {}
        for key, uploaded_file in uploaded_files.items():
            spec = DATASET_SPECS[key]
            df = read_csv_upload(uploaded_file)
            dataframes[key] = df
            if df is not None:
                with st.expander(f"{spec['label']} - {len(df)} rows", expanded=False):
                    render_dataframe_preview(spec["label"], df, spec["path"])
        
        # Ingestion controls
        st.divider()
        st.subheader("Data Ingestion")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if ingestion_locked:
                st.warning(
                    "Ingestion is locked because first-time ingestion is already completed. "
                    "You can still upload files for preview and view datasets below."
                )
            else:
                st.info(
                    "Ready to ingest. This runs only once. Data will be saved to disk and indexed into vector DBs."
                )
        with col2:
            if st.button(
                "🚀 Ingest Now",
                use_container_width=True,
                key="ingest_button",
                disabled=ingestion_locked,
            ):
                with st.spinner("Ingesting data..."):
                    try:
                        # Save uploaded files to disk
                        for key, uploaded_file in uploaded_files.items():
                            spec = DATASET_SPECS[key]
                            save_uploaded_csv(uploaded_file, spec["path"])

                        # Build dataframe set from uploaded files first, then fallback to disk
                        full_dataframes: Dict[str, Optional[pd.DataFrame]] = {}
                        for key, spec in DATASET_SPECS.items():
                            if key in dataframes and dataframes[key] is not None:
                                full_dataframes[key] = dataframes[key]
                            elif os.path.exists(spec["path"]):
                                try:
                                    full_dataframes[key] = pd.read_csv(spec["path"])
                                except Exception:
                                    full_dataframes[key] = None
                            else:
                                full_dataframes[key] = None

                        available_count = sum(1 for df in full_dataframes.values() if df is not None and not df.empty)
                        if available_count == 0:
                            st.error("No usable CSV data found for ingestion.")
                            return

                        # Ingest into vector DBs
                        counts = ingest_uploaded_dataframes(full_dataframes)

                        if not counts and has_ingestion_been_done():
                            st.warning("Ingestion already completed previously. Skipping re-ingestion.")
                            return

                        # Mark as complete
                        mark_ingestion_complete(counts)

                        st.success(f"Ingestion complete. {sum(counts.values())} documents indexed across {len(counts)} collections.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")
    else:
        if ingestion_locked:
            st.info("First-time ingestion already completed. Use the viewer below to inspect datasets.")
        else:
            st.info("Upload at least one CSV file to proceed with first-time ingestion.")
    
    st.divider()
    
    # View existing data
    st.subheader("View Existing Data")
    view_option = st.selectbox(
        "Select dataset to view",
        options=list(DATASET_SPECS.keys()),
        format_func=lambda k: DATASET_SPECS[k]["label"],
        key="view_dataset",
    )
    
    if view_option:
        spec = DATASET_SPECS[view_option]
        source_path = spec["path"]
        
        if os.path.exists(source_path):
            try:
                df = pd.read_csv(source_path)
                st.write(f"**{spec['label']}** - {len(df)} rows, {len(df.columns)} columns")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", len(df))
                col2.metric("Columns", len(df.columns))
                col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Display data
                with st.expander("View data preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
                
                # Column info
                with st.expander("Column information"):
                    col_info = []
                    for col in df.columns:
                        col_info.append({
                            "Column": col,
                            "Type": str(df[col].dtype),
                            "Non-null": df[col].notna().sum(),
                            "Unique": df[col].nunique(),
                        })
                    st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
                
                # Download option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{view_option}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_{view_option}",
                )
            except Exception as exc:
                st.error(f"Could not load {spec['label']}: {exc}")
        else:
            st.warning(f"File not found: {source_path}")
            st.info("Upload and ingest the CSV file first.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.write("")
    render_settings_panel()
    render_status_badges()
    
    sidebar_defaults = get_sidebar_defaults()
    
    with st.sidebar:
        st.header("Control Panel")
        query = st.text_area(
            "Event request",
            value=sidebar_defaults.get("query", ""),
            height=120,
            placeholder="Describe what you need help with for your event...",
        )
        category = st.text_input("Category", value=sidebar_defaults.get("category", ""), placeholder="e.g., AI Conference, Tech Summit")
        location = st.text_input("Location / Country", value=sidebar_defaults.get("location", ""), placeholder="e.g., United States, India")
        city = st.text_input("City", value=sidebar_defaults.get("city", ""), placeholder="e.g., San Francisco, Bangalore")
        audience_size = st.number_input("Audience size", min_value=0, max_value=500000, value=int(sidebar_defaults.get("audience_size", 0)), step=50)
        budget_options = ["", "low", "medium", "high", "enterprise"]
        budget_default = safe_text(sidebar_defaults.get("budget", ""))
        budget_index = budget_options.index(budget_default) if budget_default in budget_options else 0
        budget = st.selectbox("Budget", budget_options, index=budget_index)
        event_topic = st.text_input("Event topic", value=sidebar_defaults.get("event_topic", ""), placeholder="e.g., Artificial Intelligence")
        event_name = st.text_input("Event name", value=sidebar_defaults.get("event_name", ""), placeholder="e.g., TechFest 2026")
        run_clicked = st.button("Run LangGraph Planner", use_container_width=True)
    
    user_input = {
        "query": query,
        "category": category,
        "location": location,
        "city": city,
        "audience_size": int(audience_size),
        "budget": budget,
        "event_topic": event_topic,
        "event_name": event_name,
        "objective": query,
    }
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Planner",
        "Pricing",
        "Email",
        "Internal Processing",
        "LangSmith Tracing",
        "History",
        "Data",
    ])
    
    if run_clicked:
        if not query.strip():
            st.sidebar.error("Please enter an event request.")
            return
        with st.spinner("Running LangGraph planner..."):
            graph = get_graph()
            result = graph.invoke({
                "user_input": user_input,
                "query": query,
                "logs": [],
                "execution_logs": [],
                "state_flow": [],
                "quality_results": [],
                "agent_sequence": [],
                "orchestration_plan": {},
                "shared_context": "",
                "agent_outputs": {},
                "revise_count": 0,
                "rewrite_count": 0,
            })
            st.session_state["last_result"] = result
            st.session_state["last_user_input"] = user_input
            if result.get("pricing"):
                st.session_state["pricing_result"] = {"pricing": result["pricing"], "answer": result.get("sponsors_answer", "")}
            log_session_history_entry(
                query_text=query,
                final_output=safe_text(result.get("final_answer") or result.get("sponsors_answer") or result.get("answer")),
                selected_agent=safe_text(result.get("selected_agent")),
                route_target=safe_text(result.get("route_target")),
                source="planner",
            )
        st.success("Planner execution complete.")
    
    result = st.session_state.get("last_result", {})
    
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("LangGraph Event Planner")
        st.write("Multi-agent orchestration with LangGraph state machine, semantic RAG, and quality controls.")
        st.markdown("<span class='pill'>LangGraph</span><span class='pill'>Multi-agent</span><span class='pill'>Self-RAG</span><span class='pill'>Hallucination Detection</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if result:
            # Show user input context
            with st.expander("Request details", expanded=False):
                ui = st.session_state.get("last_user_input", {})
                col1, col2 = st.columns(2)
                col1.write(f"**Category**: {ui.get('category', 'N/A')}")
                col1.write(f"**Location**: {ui.get('location', 'N/A')}")
                col1.write(f"**City**: {ui.get('city', 'N/A')}")
                col2.write(f"**Audience**: {ui.get('audience_size', 'N/A'):,}")
                col2.write(f"**Budget**: {ui.get('budget', 'N/A')}")
                col2.write(f"**Event**: {ui.get('event_name', 'N/A')}")
            
            # Main output
            st.subheader("Recommendations and Analysis")
            answer = result.get("sponsors_answer", "No output generated.")
            st.markdown(answer)
            
            # Quality control metrics
            if result.get("hallucination_verdict") or result.get("usefulness_verdict"):
                with st.expander("Quality Control Metrics", expanded=False):
                    qc_col1, qc_col2 = st.columns(2)
                    qc_col1.write(f"**Hallucination Check**: {result.get('hallucination_verdict', 'N/A')}")
                    qc_col2.write(f"**Usefulness Check**: {result.get('usefulness_verdict', 'N/A')}")
                    st.write(f"**Revisions**: {result.get('revise_count', 0)} | **Rewrites**: {result.get('rewrite_count', 0)}")

            st.divider()
            render_execution_monitor(result)
        else:
            st.info("Configure event details in the Control Panel and click 'Run LangGraph Planner' to get started.")
    
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Pricing Studio")
        st.write("Analyze ticket pricing strategies, simulate different tiers, and compute break-even scenarios.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Run Pricing Engine", key="pricing_button", use_container_width=True):
            with st.spinner("Analyzing pricing..."):
                graph = get_graph()
                result = graph.invoke({
                    "user_input": user_input,
                    "query": query,
                    "selected_agent": "PRICING",
                    "logs": [],
                    "execution_logs": [],
                    "state_flow": [],
                    "quality_results": [],
                    "agent_sequence": [],
                    "orchestration_plan": {},
                    "shared_context": "",
                    "agent_outputs": {},
                    "revise_count": 0,
                    "rewrite_count": 0,
                })
                st.session_state["pricing_result"] = {"pricing": result.get("pricing"), "answer": result.get("sponsors_answer", "")}
                log_session_history_entry(
                    query_text=query,
                    final_output=safe_text(result.get("final_answer") or result.get("sponsors_answer") or result.get("answer")),
                    selected_agent=safe_text(result.get("selected_agent")),
                    route_target=safe_text(result.get("route_target")),
                    source="pricing",
                )
            st.success("Pricing analysis complete.")
        
        pricing_result = st.session_state.get("pricing_result")
        if pricing_result:
            # Strategy and insights
            answer = pricing_result.get("answer", "")
            st.subheader("Strategy and Insights")
            st.markdown(answer)
            
            st.divider()
            
            # Pricing simulator
            st.subheader("🎮 Ticket Tier Simulator")
            st.write("Adjust prices, capacities, and conversion rates to simulate revenue impact.")
            render_pricing_simulator(pricing_result.get("pricing", pricing_result))
            
            st.divider()
            
            # Break-even analysis
            st.subheader("Break-Even Analysis")
            st.write("Calculate minimum tickets needed to cover costs and feasibility timeline.")
            render_break_even_analysis(pricing_result, user_input)

            st.divider()

            # Revenue and conversion forecasting
            render_revenue_conversion_forecasting(pricing_result, user_input)
        else:
            st.info("Click 'Run Pricing Engine' to analyze pricing scenarios.")
    
    with tab3:
        render_email_outreach_tab(user_input, result)

    with tab4:
        render_internal_processing_visualization(result, st.session_state.get("pricing_result", {}))

    with tab5:
        render_langsmith_tracing_tab()

    with tab6:
        render_history_tab()

    with tab7:
        render_csv_upload_interface()


if __name__ == "__main__":
    main()
