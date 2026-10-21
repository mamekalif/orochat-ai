import uuid
from datetime import datetime
import streamlit as st
from groq import Groq
from supabase import create_client, Client

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Orochat-ai 🇪🇹", 
    page_icon="🇪🇹", 
    layout="wide"
)

# Custom Styling for Modern Dark UI & Centered Layout
st.markdown("""
<style>
    .stSidebar {
        background-color: #0d0f12;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    /* Style chat input container */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# API Clients & Database Connections
# ---------------------------------------------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

client = get_groq_client()

try:
    supabase = get_supabase_client()
except Exception:
    supabase = None

# ---------------------------------------------------------
# Dynamic Model Fetcher (Permanent 404 Prevention)
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def get_active_groq_model():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        
        preferred_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]
        
        for p_model in preferred_models:
            if p_model in model_ids:
                return p_model
                
        for m_id in model_ids:
            if not any(x in m_id.lower() for x in ["whisper", "guard", "vision", "embed"]):
                return m_id
                
        return "llama-3.1-8b-instant"
    except Exception:
        return "llama-3.1-8b-instant"

ACTIVE_MODEL = get_active_groq_model()

# ---------------------------------------------------------
# Session & State Management
# ---------------------------------------------------------
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "Haasaa Haarawaa"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None

def start_new_chat():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.current_chat_title = "Haasaa Haarawaa"
    st.session_state.messages = []
    st.session_state.last_interaction_id = None

def load_past_chat(session_id, chat_title):
    st.session_state.current_session_id = session_id
    st.session_state.current_chat_title = chat_title
    st.session_state.messages = []
    
    if supabase:
        try:
            res = supabase.table("chat_logs") \
                .select("user_prompt, ai_response") \
                .eq("session_id", session_id) \
                .order("created_at", desc=False) \
                .execute()
            
            for row in res.data:
                st.session_state.messages.append({"role": "user", "content": row["user_prompt"]})
                st.session_state.messages.append({"role": "assistant", "content": row["ai_response"]})
        except Exception:
            pass

def log_to_supabase(user_prompt, ai_response, session_id, chat_title):
    if supabase:
        try:
            data = {
                "user_prompt": user_prompt,
                "ai_response": ai_response,
                "model_used": ACTIVE_MODEL,
                "session_id": session_id,
                "chat_title": chat_title
            }
            res = supabase.table("chat_logs").insert(data).execute()
            if res.data:
                return res.data[0]["id"]
        except Exception:
            pass
    return None

def update_feedback(log_id, feedback_value):
    if supabase and log_id:
        try:
            supabase.table("chat_logs").update({"feedback": feedback_value}).eq("id", log_id).execute()
        except Exception:
            pass

# ---------------------------------------------------------
# Sidebar Navigation (New Chat & Recents)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇪🇹 Orochat-ai")
    
    if st.button("➕ Haasaa Haarawaa (New Chat)", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Seenaa Haasaa (Recents)")

    if supabase:
        try:
            response = supabase.table("chat_logs") \
                .select("session_id, chat_title, created_at") \
                .order("created_at", desc=True) \
                .execute()
            
            seen_sessions = set()
            recent_chats = []
            for item in response.data:
                s_id = item.get("session_id")
                if s_id and s_id not in seen_sessions:
                    seen_sessions.add(s_id)
                    recent_chats.append(item)

            for chat in recent_chats[:15]:
                title = chat.get("chat_title") or "Haasaa"
                sid = chat.get("session_id")
                is_active = (sid == st.session_state.current_session_id)
                btn_label = f"💬 {title[:25]}..." if len(title) > 25 else f"💬 {title}"
                
                if st.button(btn_label, key=f"session_{sid}", use_container_width=True, disabled=is_active):
                    load_past_chat(sid, title)
                    st.rerun()
        except Exception:
            pass

# ---------------------------------------------------------
# UI Layout: Modern Landing Page vs Active Chat Canvas
# ---------------------------------------------------------
if not st.session_state.messages:
    # Sleek, Centered Landing Page
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 18vh;'>
            <div style='font-size: 3.5rem; margin-bottom: 10px;'>✨</div>
            <h1 style='color: #ffffff; font-size: 2.8rem; font-weight: 600; margin-bottom: 5px;'>Orochat-ai</h1>
            <p style='color: #9aa0a6; font-size: 1.2rem; margin-bottom: 30px;'>Ready when you are / Yeroo hundumaa qophii dha</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Active Conversation Header
    st.subheader(f"💬 {st.session_state.current_chat_title}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Feedback Controls for the last response
    if st.session_state.last_interaction_id:
        st.caption("Deebii kana irratti yaada kee kenni:")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Deebii Gaarii", key="fb_pos"):
                update_feedback(st.session_state.last_interaction_id, "positive")
                st.toast("Galatoomaa! Yaadni keessan galmeeffameera.")
                st.session_state.last_interaction_id = None
        with col2:
            if st.button("👎 Deebii Dogoggoraa", key="fb_neg"):
                update_feedback(st.session_state.last_interaction_id, "negative")
                st.toast("Galatoomaa! Sirreessuuf data keessatti galmeessineerra.")
                st.session_state.last_interaction_id = None

# ---------------------------------------------------------
# Advanced System Prompt & Chat Processing Pipeline
# ---------------------------------------------------------
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

SYSTEM_PROMPT = f"""Ati Orochat-ai dha. Gargaaraa teeknoolojii AI fi Afaan Oromoo ti, kan Mohammed Kalif Umar uume.
Seera Deebii fi Qulqullina Loli (Vocabulary & Grammar Rules):
1. Yeroo hunda Afaan Oromoo qulqulluu, uumamaa (natural), fi sirrii ta'een deebisi.
2. Hiika jechoota aadaa, hiika jechoota diizinarii, akkasumas kontekstii naannoo (fkn: Harargee) sirriitti hubachuun ibsi.
3. Yeroo fi guyyaan har'aa: {current_time_str} (EAT). Gaaffii sa'aatii ykn yeroo dhiyaatu kana irratti hundaa'i.
"""

if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    if not st.session_state.messages or st.session_state.current_chat_title == "Haasaa Haarawaa":
        st.session_state.current_chat_title = prompt[:30]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Append recent chat turns for short-term conversational memory
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=payload,
                    temperature=0.6,
                    max_tokens=600,
                    stream=True
                )

                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )

                st.session_state.messages.append({"role": "assistant", "content": reply})

                log_id = log_to_supabase(
                    user_prompt=prompt,
                    ai_response=reply,
                    session_id=st.session_state.current_session_id,
                    chat_title=st.session_state.current_chat_title
                )
                st.session_state.last_interaction_id = log_id
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
