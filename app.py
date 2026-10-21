import uuid
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

# Custom Styling for Sidebar UI
st.markdown("""
<style>
    .stSidebar {
        background-color: #0d0f12;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
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
# Dynamic Model Loader
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_active_groq_model():
    try:
        available_models = client.models.list()
        model_ids = [m.id for m in available_models.data]
        priority_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b"
        ]
        for model in priority_models:
            if model in model_ids:
                return model
        for m_id in model_ids:
            if not any(x in m_id.lower() for x in ["whisper", "guard", "vision", "audio"]):
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

# Helper function to start a new chat session
def start_new_chat():
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.current_chat_title = "Haasaa Haarawaa"
    st.session_state.messages = []
    st.session_state.last_interaction_id = None

# Helper function to load a past chat session from Supabase
def load_past_chat(session_id, chat_title):
    st.session_state.current_session_id = session_id
    st.session_state.current_chat_title = chat_title
    st.session_state.messages = []
    
    if supabase:
        try:
            res = supabase.table("chat_logs") \
                .select("user_prompt, ai_response, id") \
                .eq("session_id", session_id) \
                .order("created_at", desc=False) \
                .execute()
            
            for row in res.data:
                st.session_state.messages.append({"role": "user", "content": row["user_prompt"]})
                st.session_state.messages.append({"role": "assistant", "content": row["ai_response"]})
        except Exception as e:
            st.error(f"Error loading chat history: {e}")

# Database Logging Helper
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
# RAG Knowledge Engine & System Instructions
# ---------------------------------------------------------
KNOWLEDGE_BASE = {
    "mammaaksa": "Mammaaksa: 'Haadha manaa fi abbaa manaa walmormu, alagaatu gidduutti fayyadama.'",
    "baabbile": "Magaalaan Baabbilee Godina Bahaa Harargee keessatti argamti. Naannoo bosona leencotaa fi Arbaatiin beekamti.",
    "orochat": "Orochat-ai gargaaraa teeknoolojii AI fi Afaan Oromoo kan Mohammed Kalif Umar tu ijaareedha.",
    "mohammed": "Mohammed Kalif Umar kalaqaa fi ijaaraa Orochat-ai fi teeknoolojii Afaan Oromoo ti."
}

def retrieve_knowledge(query):
    q = query.lower()
    retrieved = [fact for key, fact in KNOWLEDGE_BASE.items() if key in q]
    return "\n".join(retrieved) if retrieved else None

SYSTEM_PROMPT = """Ati Orochat-ai dha. Ati gargaaraa teeknoolojii AI fi Afaan Oromoo ti.

Eenyummaa fi Kalaqaa Keeti:
- Maqaan kee: Orochat-ai
- Namni si uume/kalaqe: Mohammed Kalif Umar
- Kaayyoon kee: Ummata Oromootaaf teeknoolojii AI fi Afaan Oromoo tajaajiluu.

Seera Deebii:
1. Deebii kee hunda Afaan Oromoo qulqulluu, sirrii, fi uumamaan (conversational) ta'een deebisi.
2. Gaaffii "Eenyuutu si uume?", "Eenyuutu si kalaqe?" jedhuuf: "Ani Orochat-ai dha. Mohammed Kalif Umar tu na kalaqe." jedhi.
3. Yoo odeeffannoo dabalataa (Knowledge Context) siif keenname, odeeffannoo sana fayyadamii deebii qulqulluu kenni.
"""

# ---------------------------------------------------------
# Sidebar Navigation (New Chat & Recents)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇪🇹 Orochat-ai")
    
    # 📝 New Chat Button
    if st.button("➕ Haasaa Haarawaa (New Chat)", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Seenaa Haasaa (Recents)")

    # Load unique sessions from database
    if supabase:
        try:
            response = supabase.table("chat_logs") \
                .select("session_id, chat_title, created_at") \
                .order("created_at", desc=True) \
                .execute()
            
            # Deduplicate by session_id while maintaining order
            seen_sessions = set()
            recent_chats = []
            for item in response.data:
                s_id = item.get("session_id")
                if s_id and s_id not in seen_sessions:
                    seen_sessions.add(s_id)
                    recent_chats.append(item)

            # Display past sessions in sidebar list
            for chat in recent_chats[:15]:  # Display last 15 chats
                title = chat.get("chat_title") or "Haasaa"
                sid = chat.get("session_id")
                
                # Active highlight indicator
                is_active = (sid == st.session_state.current_session_id)
                btn_label = f"💬 {title[:28]}..." if len(title) > 28 else f"💬 {title}"
                
                if st.button(btn_label, key=f"session_{sid}", use_container_width=True, disabled=is_active):
                    load_past_chat(sid, title)
                    st.rerun()

        except Exception as e:
            st.caption("Seenaa haasaa fe'uu irratti rakkootu uumame.")
    else:
        st.caption("Database hin tengiinsifamne.")

# ---------------------------------------------------------
# Main Chat Window
# ---------------------------------------------------------
st.subheader(f"💬 {st.session_state.current_chat_title}")

# Render chat message history for current active session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Feedback Controls
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
# Chat Input & Streaming Pipeline
# ---------------------------------------------------------
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    # Set chat title automatically from first question if it's new
    if not st.session_state.messages or st.session_state.current_chat_title == "Haasaa Haarawaa":
        st.session_state.current_chat_title = prompt[:30]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Retrieve context from RAG
    context = retrieve_knowledge(prompt)
    if context:
        payload.append({"role": "system", "content": f"Odeeffannoo Dabalataa:\n{context}"})

    # Append short-term session conversation context
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=payload,
                    temperature=0.6,
                    max_tokens=400,
                    stream=True
                )

                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )

                st.session_state.messages.append({"role": "assistant", "content": reply})

                # Save chat to Supabase database
                log_id = log_to_supabase(
                    user_prompt=prompt,
                    ai_response=reply,
                    session_id=st.session_state.current_session_id,
                    chat_title=st.session_state.current_chat_title
                )
                
                st.session_state.last_interaction_id = log_id
                st.rerun()

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
