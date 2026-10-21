import uuid
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from groq import Groq
from supabase import create_client, Client
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Orochat-ai 🇪🇹",
    page_icon="🇪🇹",
    layout="wide"
)

st.markdown("""
<style>
    .stSidebar { background-color: #0d0f12; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# API Clients
# ---------------------------------------------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

client = get_groq_client()
try:
    supabase = get_supabase_client()
except Exception:
    supabase = None

# ---------------------------------------------------------
# Active Model
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def get_active_groq_model():
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        for pref in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                     "llama3-70b-8192", "llama3-8b-8192"]:
            if pref in ids:
                return pref
        for m in ids:
            if not any(x in m.lower() for x in ["whisper", "guard", "vision", "embed"]):
                return m
    except Exception:
        pass
    return "llama-3.1-8b-instant"

ACTIVE_MODEL = get_active_groq_model()

# ---------------------------------------------------------
# LEARNING LAYER — the core of the new AI
# ---------------------------------------------------------
DATA_DIR = Path("mammaaksa_ai_data")

@st.cache_resource
def load_static_corpus():
    """Load the static Mammaaksa proverbs once."""
    try:
        return json.load(open(DATA_DIR / "proverb_corpus.json", encoding="utf-8"))
    except FileNotFoundError:
        # Fallback: try the original file
        try:
            raw = json.load(open("Mammaaksa.json", encoding="utf-8"))
            return [{
                "id": p["ID"],
                "proverb": p["Mammaaksa_AO"],
                "meaning_om": p.get("Meaning_AO", ""),
                "meaning_en": p.get("Meaning_EN", ""),
            } for p in raw]
        except Exception:
            return []

def _fetch_from_supabase(table, limit=2000):
    """Safely fetch rows from Supabase, return [] on failure."""
    if not supabase:
        return []
    try:
        return supabase.table(table).select("*").limit(limit).execute().data or []
    except Exception:
        return []

def build_learning_index():
    """
    Combine 4 sources into ONE BM25 index.
    Rebuilds only when learning_version changes.
    """
    static   = load_static_corpus()
    kb_rows  = _fetch_from_supabase("knowledge_base")
    mem_rows = _fetch_from_supabase("user_memory")

    docs = []

    # Priority 1: corrections + manual facts (from knowledge_base)
    for r in kb_rows:
        text = f'{r["question"]} {r["answer"]}'
        docs.append({
            "text": text,
            "source": r["category"],
            "priority": r["priority"],
            "payload": r,
        })

    # Priority 3: static mammaaksa corpus
    for p in static:
        text = f'{p.get("proverb","")} {p.get("meaning_om","")} {p.get("meaning_en","")}'
        docs.append({
            "text": text,
            "source": "mammaaksa",
            "priority": 5,
            "payload": p,
        })

    # User memory (facts) — treated as very high priority
    for m in mem_rows:
        docs.append({
            "text": m["fact"],
            "source": "memory",
            "priority": 1,
            "payload": m,
        })

    if not docs:
        return None

    tokens = [re.findall(r"\w+", d["text"].lower()) for d in docs]
    bm25   = BM25Okapi(tokens)
    return {"docs": docs, "bm25": bm25, "size": len(docs)}

def get_learning_index():
    """Rebuild when version changes, else reuse."""
    version = st.session_state.get("learning_version", 0)
    if st.session_state.get("_idx_version") != version:
        st.session_state._idx_version = version
        st.session_state._index = build_learning_index()
    return st.session_state.get("_index")

def retrieve_context(query, k=5):
    """Return a markdown context block from the learning index."""
    idx = get_learning_index()
    if not idx:
        return ""

    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return ""

    scores = idx["bm25"].get_scores(tokens)
    # Rank: BM25 score, tie-broken by priority (lower priority = more important)
    ranked = sorted(
        range(len(idx["docs"])),
        key=lambda i: (scores[i], -idx["docs"][i]["priority"]),
        reverse=True,
    )[:k]

    lines = []
    for i in ranked:
        if scores[i] <= 0.0:
            continue
        d = idx["docs"][i]

        if d["source"] == "mammaaksa":
            p = d["payload"]
            lines.append(f'- **"{p["proverb"]}"** — {p.get("meaning_en","")} (Oromoo: {p.get("meaning_om","")})')
        elif d["source"] == "memory":
            lines.append(f'- MEMORY: {d["payload"]["fact"]}')
        else:
            r = d["payload"]
            label = {"correction": "CORRECTION", "manual": "NOTE"}.get(r["category"], "KNOWN")
            lines.append(f'- {label} Q: {r["question"]}\n  A: {r["answer"]}')

    if not lines:
        return ""
    return "### CONTEXT (GROUND TRUTH)\n" + "\n".join(lines) + "\n### END CONTEXT"

# ---------------------------------------------------------
# WRITE-BACK: learning actions
# ---------------------------------------------------------
def bump_learning_version():
    st.session_state.learning_version = st.session_state.get("learning_version", 0) + 1

def save_verified(question, answer):
    if supabase:
        try:
            supabase.table("knowledge_base").insert({
                "question": question,
                "answer": answer,
                "category": "verified",
                "priority": 3,
            }).execute()
            bump_learning_version()
            return True
        except Exception:
            return False
    return False

def save_correction(question, wrong_answer, correct_answer):
    if supabase:
        try:
            supabase.table("knowledge_base").insert({
                "question": question,
                "answer": correct_answer,
                "category": "correction",
                "priority": 1,
            }).execute()
            bump_learning_version()
            return True
        except Exception:
            return False
    return False

def save_manual_note(question, answer):
    if supabase:
        try:
            supabase.table("knowledge_base").insert({
                "question": question,
                "answer": answer,
                "category": "manual",
                "priority": 2,
            }).execute()
            bump_learning_version()
            return True
        except Exception:
            return False
    return False

def save_memory(fact):
    if supabase:
        try:
            supabase.table("user_memory").insert({"fact": fact}).execute()
            bump_learning_version()
            return True
        except Exception:
            return False
    return False

# ---------------------------------------------------------
# Session & Chat State
# ---------------------------------------------------------
defaults = {
    "current_session_id": str(uuid.uuid4()),
    "current_chat_title": "Haasaa Haarawaa",
    "messages": [],
    "last_interaction_id": None,
    "learning_version": 0,
    "pending_negative": None,   # for the correction textbox
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

def start_new_chat():
    st.session_state.current_session_id  = str(uuid.uuid4())
    st.session_state.current_chat_title  = "Haasaa Haarawaa"
    st.session_state.messages            = []
    st.session_state.last_interaction_id = None
    st.session_state.pending_negative    = None

def load_past_chat(session_id, title):
    st.session_state.current_session_id  = session_id
    st.session_state.current_chat_title  = title
    st.session_state.messages            = []
    if supabase:
        try:
            res = (supabase.table("chat_logs")
                   .select("user_prompt, ai_response")
                   .eq("session_id", session_id)
                   .order("created_at", desc=False)
                   .execute())
            for row in res.data:
                st.session_state.messages.append({"role": "user",      "content": row["user_prompt"]})
                st.session_state.messages.append({"role": "assistant", "content": row["ai_response"]})
        except Exception:
            pass

def log_chat(user_prompt, ai_response, session_id, title):
    if supabase:
        try:
            res = supabase.table("chat_logs").insert({
                "user_prompt": user_prompt,
                "ai_response": ai_response,
                "model_used":  ACTIVE_MODEL,
                "session_id":  session_id,
                "chat_title":  title,
            }).execute()
            return res.data[0]["id"] if res.data else None
        except Exception:
            return None
    return None

def update_feedback(log_id, value):
    if supabase and log_id:
        try:
            supabase.table("chat_logs").update({"feedback": value}).eq("id", log_id).execute()
        except Exception:
            pass

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇪🇹 Orochat-ai")

    if st.button("➕ Haasaa Haarawaa", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Seenaa (Recents)")

    if supabase:
        try:
            res = (supabase.table("chat_logs")
                   .select("session_id, chat_title, created_at")
                   .order("created_at", desc=True)
                   .execute())
            seen = set()
            for item in res.data:
                sid = item.get("session_id")
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                title = item.get("chat_title") or "Haasaa"
                active = (sid == st.session_state.current_session_id)
                label = f"💬 {title[:25]}..." if len(title) > 25 else f"💬 {title}"
                if st.button(label, key=f"s_{sid}", use_container_width=True, disabled=active):
                    load_past_chat(sid, title)
                    st.rerun()
                if len(seen) >= 12:
                    break
        except Exception:
            pass

    # ----- LEARNING STATS (visible proof it grows) -----
    st.markdown("---")
    st.subheader("📈 Barnoota (Learning)")
    idx = get_learning_index()
    if idx:
        sources = {}
        for d in idx["docs"]:
            sources[d["source"]] = sources.get(d["source"], 0) + 1
        st.metric("Knowledge units", idx["size"])
        for src, cnt in sorted(sources.items()):
            st.caption(f"• {src}: **{cnt}**")
    else:
        st.caption("⚠️ No learning data yet")

    # ----- MANUAL TEACHING (feed it directly) -----
    with st.expander("➕ Barsiisi (Teach AI)"):
        kind = st.radio("Gosa", ["Note", "Memory"], horizontal=True, label_visibility="collapsed")

        if kind == "Note":
            q = st.text_input("Gaaffii (Question)", key="teach_q")
            a = st.text_area("Deebii (Answer)", key="teach_a", height=80)
            if st.button("💾 Galmeessi", key="teach_note") and q and a:
                if save_manual_note(q, a):
                    st.success("✅ Dabalaman!")
                    st.rerun()
        else:
            fact = st.text_input("Fact (e.g. 'User's name is Mohammed')", key="teach_mem")
            if st.button("🧠 Yaadadhu", key="teach_mem_btn") and fact:
                if save_memory(fact):
                    st.success("✅ Yaadame!")
                    st.rerun()

# ---------------------------------------------------------
# Landing vs Chat Canvas
# ---------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;margin-top:18vh;'>
        <div style='font-size:3.5rem;margin-bottom:10px;'>✨</div>
        <h1 style='color:#fff;font-size:2.8rem;font-weight:600;margin-bottom:5px;'>Orochat-ai</h1>
        <p style='color:#9aa0a6;font-size:1.2rem;margin-bottom:30px;'>
        Ready when you are / Yeroo hundumaa qophiidha</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.subheader(f"💬 {st.session_state.current_chat_title}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------- FEEDBACK / LEARNING BUTTONS ----------------
    if st.session_state.last_interaction_id:
        st.caption("Deebii kana irratti yaada kee kenni:")

        # Find the Q&A pair that produced the last response
        last_user = next((m["content"] for m in reversed(st.session_state.messages)
                          if m["role"] == "user"), "")
        last_bot  = next((m["content"] for m in reversed(st.session_state.messages)
                          if m["role"] == "assistant"), "")

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Gaarii", key="fb_pos"):
                update_feedback(st.session_state.last_interaction_id, "positive")
                if save_verified(last_user, last_bot):
                    st.toast("✅ AI baratte! (AI learned this)")
                else:
                    st.toast("👍 Galatoomaa!")
                st.session_state.last_interaction_id = None
                st.rerun()

        with col2:
            if st.button("👎 Dogoggora", key="fb_neg"):
                update_feedback(st.session_state.last_interaction_id, "negative")
                st.session_state.pending_negative = {
                    "q": last_user, "wrong": last_bot
                }
                st.session_state.last_interaction_id = None
                st.rerun()

        # Correction textbox (only appears after 👎)
        if st.session_state.get("pending_negative"):
            p = st.session_state.pending_negative
            st.warning(f"Deebii sirrii qoricha: **{p['q']}**")
            correction = st.text_area(
                "Deebii sirrii barreessi (teach the correct answer):",
                key="correction_input", height=100
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("💾 Sirreessi (Save correction)", key="save_corr") and correction:
                    if save_correction(p["q"], p["wrong"], correction):
                        st.success("✅ AI sirreeffame! (AI corrected)")
                        st.session_state.pending_negative = None
                        st.rerun()
            with c2:
                if st.button("❌ Dhiisi", key="cancel_corr"):
                    st.session_state.pending_negative = None
                    st.rerun()

# ---------------------------------------------------------
# Chat Handler
# ---------------------------------------------------------
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

BASE_SYSTEM = f"""Ati Orochat-ai dha. Gargaaraa teeknoolojii AI fi Afaan Oromoo ti, kan Mohammed Kalif Umar uume.
Seera Deebii:
1. Afaan Oromoo qulqulluu, uumamaa fi sirrii ta'een deebisi.
2. Hiika aadaa, hiika diizinarii, akkasumas naannoo (fkn: Harargee) hubadhu.
3. Yeroo har'aa: {current_time_str} (EAT).
4. Yoo CONTEXT armaan gadii keessatti mammaaksa, qajeelfama, ykn yaadannoo jiru, ISA IRRATTI HUNDAA'I.
   Kanneen CONTEXT keessaa mirkanaa'oo fi dursa qabu.
"""

def build_system_prompt(user_query: str) -> str:
    ctx = retrieve_context(user_query)
    return BASE_SYSTEM + ("\n\n" + ctx if ctx else "")

if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    if not st.session_state.messages or st.session_state.current_chat_title == "Haasaa Haarawaa":
        st.session_state.current_chat_title = prompt[:30]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build payload: system + last 6 turns
    payload = [{"role": "system", "content": build_system_prompt(prompt)}]
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=payload,
                    temperature=0.5,
                    max_tokens=700,
                    stream=True,
                )
                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})

                log_id = log_chat(
                    user_prompt=prompt,
                    ai_response=reply,
                    session_id=st.session_state.current_session_id,
                    title=st.session_state.current_chat_title,
                )
                st.session_state.last_interaction_id = log_id

                # Show a small debug expander (optional — remove in prod)
                with st.expander("🔍 Mammaaksa argame", expanded=False):
                    st.code(retrieve_context(prompt) or "—")

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
