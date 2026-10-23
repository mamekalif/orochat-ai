import uuid, json, math, re
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from groq import Groq
from supabase import create_client

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Orochat-ai 🇪🇹", page_icon="🇪🇹", layout="wide")

# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------
@st.cache_resource
def _groq():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def _sb():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None

client = _groq()
supabase = _sb()

# ---------------------------------------------------------
# Active Groq model (2026: GPT-OSS replaces retired Llama)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def pick_model():
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        return "openai/gpt-oss-20b"

    # Groq's recommended replacements for retired Llama models (2026-08-16)
    for pref in [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]:
        if pref in ids:
            return pref

    # Last resort: any gpt-oss model available
    gpt_oss = [m for m in ids if "gpt-oss" in m.lower()]
    return gpt_oss[0] if gpt_oss else "openai/gpt-oss-20b"

ACTIVE_MODEL = pick_model()

# ---------------------------------------------------------
# Load corpus (repo root)
# ---------------------------------------------------------
@st.cache_resource
def load_corpus():
    for name in ["proverb_corpus.json", "Mammaaksa.json", "qa_pairs.json"]:
        p = Path(name)
        if not p.exists():
            continue
        try:
            raw = json.load(open(p, encoding="utf-8"))

            # Original Mammaaksa.json structure
            if name == "Mammaaksa.json":
                return [{
                    "id": x["ID"],
                    "proverb": x.get("Mammaaksa_AO", ""),
                    "meaning_om": x.get("Meaning_AO", ""),
                    "meaning_en": x.get("Meaning_EN", ""),
                } for x in raw]

            # qa_pairs.json fallback (extract meaning rows only)
            if name == "qa_pairs.json":
                out, seen = [], set()
                for q in raw:
                    if q.get("type") != "meaning":
                        continue
                    k = q.get("proverb_ao", "")
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    out.append({
                        "id": q.get("source_id", ""),
                        "proverb": k,
                        "meaning_om": q.get("answer_om", ""),
                        "meaning_en": q.get("answer_en", ""),
                    })
                return out

            # proverb_corpus.json (already compact)
            return raw
        except Exception:
            continue
    return []

CORPUS = load_corpus()

# ---------------------------------------------------------
# Pure-Python BM25 (no external deps)
# ---------------------------------------------------------
def tok(t: str):
    return re.findall(r"\w+", t.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.N = len(self.docs)
        self.avg = sum(len(d) for d in self.docs) / self.N if self.N else 0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for c in self.tf:
            for t in c:
                df[t] += 1
        self.idf = {
            t: math.log((self.N - n + 0.5) / (n + 0.5) + 1)
            for t, n in df.items()
        }

    def scores(self, q):
        out = [0.0] * self.N
        for term in tok(q):
            if term not in self.idf:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg)
                out[i] += self.idf[term] * (f * (self.k1 + 1)) / denom
        return out

CORPUS_TXT = [
    f'{p["proverb"]} {p.get("meaning_om", "")} {p.get("meaning_en", "")}'
    for p in CORPUS
]
BM = BM25(CORPUS_TXT) if CORPUS else None

# ---------------------------------------------------------
# Supabase learning layer
# ---------------------------------------------------------
def _fetch(table, lim=2000):
    if not supabase:
        return []
    try:
        return supabase.table(table).select("*").limit(lim).execute().data or []
    except Exception:
        return []

def retrieve(q, k=4):
    lines = []

    # 1) User memory (facts)
    q_tokens = set(tok(q))
    for r in _fetch("user_memory", 500):
        if q_tokens & set(tok(r.get("fact", ""))):
            lines.append(f'- MEMORY: {r["fact"]}')

    # 2) Learned notes & corrections
    for r in _fetch("knowledge_base", 500):
        txt = r.get("question", "") + " " + r.get("answer", "")
        if q_tokens & set(tok(txt)):
            label = "CORRECTION" if r.get("category") == "correction" else "NOTE"
            lines.append(f'- {label}: {r.get("question","")} → {r.get("answer","")}')

    # 3) Proverbs (BM25)
    if BM:
        sc = BM.scores(q)
        for i in sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:k]:
            if sc[i] <= 0:
                continue
            p = CORPUS[i]
            lines.append(
                f'- "{p["proverb"]}" — {p.get("meaning_en","")} '
                f'(Oromoo: {p.get("meaning_om","")})'
            )

    return "### CONTEXT\n" + "\n".join(lines[:8]) + "\n### END" if lines else ""

def learn_note(q, a, cat="manual", pri=2):
    if supabase:
        try:
            supabase.table("knowledge_base").insert({
                "question": q,
                "answer": a,
                "category": cat,
                "priority": pri,
            }).execute()
        except Exception:
            pass

def learn_mem(f):
    if supabase:
        try:
            supabase.table("user_memory").insert({"fact": f}).execute()
        except Exception:
            pass

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
for k, v in {
    "sid": str(uuid.uuid4()),
    "title": "Haasaa Haarawaa",
    "msgs": [],
    "last_q": "",
    "last_a": "",
    "pending_neg": None,
}.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇪🇹 Orochat-ai")

    if st.button("➕ Haasaa Haarawaa", use_container_width=True, type="primary"):
        st.session_state.sid = str(uuid.uuid4())
        st.session_state.title = "Haasaa Haarawaa"
        st.session_state.msgs = []
        st.session_state.pending_neg = None
        st.rerun()

    st.markdown("---")
    st.subheader("📈 Barnoota (Learning)")
    st.metric("Proverbs", len(CORPUS))
    st.caption(f"Notes: {len(_fetch('knowledge_base', 5000))}")
    st.caption(f"Memory: {len(_fetch('user_memory', 5000))}")

    with st.expander("➕ Barsiisi (Teach)"):
        kind = st.radio("Type", ["Note", "Memory"], horizontal=True)
        if kind == "Note":
            q = st.text_input("Question", key="tq")
            a = st.text_area("Answer", height=60, key="ta")
            if st.button("💾 Save Note") and q and a:
                learn_note(q, a)
                st.success("Saved!")
                st.rerun()
        else:
            f = st.text_input("Fact", key="tm")
            if st.button("🧠 Save Memory") and f:
                learn_mem(f)
                st.success("Saved!")
                st.rerun()

# ---------------------------------------------------------
# Landing vs Chat
# ---------------------------------------------------------
if not st.session_state.msgs:
    st.markdown("""
    <div style='text-align:center;margin-top:18vh;'>
      <div style='font-size:3.5rem;'>✨</div>
      <h1>Orochat-ai</h1>
      <p style='color:#9aa0a6;'>Ready when you are / Yeroo hundumaa qophiidha</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.subheader(f"💬 {st.session_state.title}")
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Feedback buttons
    if st.session_state.last_q and not st.session_state.pending_neg:
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("👍 Gaarii"):
                learn_note(st.session_state.last_q, st.session_state.last_a, "verified", 3)
                st.toast("✅ AI baratte!")
                st.rerun()
        with c2:
            if st.button("👎 Dogoggora"):
                st.session_state.pending_neg = {
                    "q": st.session_state.last_q,
                    "a": st.session_state.last_a,
                }
                st.rerun()

    # Correction box (after 👎)
    if st.session_state.pending_neg:
        p = st.session_state.pending_neg
        st.warning(f"Correct: **{p['q']}**")
        fixed = st.text_area("Correct answer", key="fix", height=70)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("💾 Save correction") and fixed:
                learn_note(p["q"], fixed, "correction", 1)
                st.success("Corrected!")
                st.session_state.pending_neg = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.pending_neg = None
                st.rerun()

# ---------------------------------------------------------
# Chat input & generation
# ---------------------------------------------------------
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
BASE_SYSTEM = (
    f"Ati Orochat-ai dha. Gargaaraa Afaan Oromoo fi mammaaksa.\n"
    f"Yeroo har'aa: {now} (EAT).\n"
    f"Yoo CONTEXT jiru, ISA IRRATTI HUNDAA'I. Deebii kee Afaan Oromoo qulqulluun kenni."
)

if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    if not st.session_state.msgs:
        st.session_state.title = prompt[:30]

    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ctx = retrieve(prompt)
    sys_prompt = BASE_SYSTEM + ("\n\n" + ctx if ctx else "")
    payload = [{"role": "system", "content": sys_prompt}] + st.session_state.msgs[-6:]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=payload,
                temperature=0.5,
                max_tokens=700,
                stream=True,
                # GPT-OSS reasoning control (via extra_body)
                extra_body={
                    "include_reasoning": False,
                    "reasoning_effort": "low",
                },
            )
            reply = st.write_stream(
                c.choices[0].delta.content or "" for c in stream
            )
            st.session_state.msgs.append({"role": "assistant", "content": reply})
            st.session_state.last_q = prompt
            st.session_state.last_a = reply
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
