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

# Custom Styling
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
# API Clients & Database
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
# Authentication UI & Session Handling
# ---------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

def render_auth_page():
    st.title("🇪🇹 Orochat-ai - Login / Register")
    st.write("Akkam! Orochat-ai fayyadamuuf mee dura seenaa (Log in).")

    tab1, tab2 = st.tabs(["🔑 Log In", "📝 Register"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", type="primary", use_container_width=True):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = res.user
                st.success("Milkaa'inaan seentaniittu!")
                st.rerun()
            except Exception as e:
                st.error(f"Error logging in: {e}")

    with tab2:
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register", use_container_width=True):
            try:
                res = supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                st.success("Galmeen milkaa'eera! Email keessan mirkaneessaa.")
            except Exception as e:
                st.error(f"Error registering: {e}")

    st.markdown("---")
    st.subheader("Or continue with")
    if st.button("🌐 Continue with Google (Gmail)", use_container_width=True):
        try:
            res = supabase.auth.sign_in_with_oauth({"provider": "google"})
            st.info(f"Redirecting to Google auth... {res.url}")
        except Exception as e:
            st.error(f"Google login error: {e}")

# Block app unless logged in
if not st.session_state.user:
    render_auth_page()
    st.stop()

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
    if supabase and st.session_state.user:
        try:
            data = {
                "user_prompt": user_prompt,
                "ai_response": ai_response,
                "model_used": "llama-3.3-70b-versatile",
                "session_id": session_id,
                "chat_title": chat_title,
                "user_id": st.session_state.user.id
            }
            res = supabase.table("chat_logs").insert(data).execute()
            if res.data:
                return res.data[0]["id"]
        except Exception:
            pass
    return None

# ---------------------------------------------------------
# Sidebar Navigation (User Profile, New Chat & Recents)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🇪🇹 Orochat-ai")
    
    # Display Active User Email
    user_email = st.session_state.user.email if st.session_state.user else "User"
    st.caption(f"👤 {user_email}")
    if st.button("🚪 Logout", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

    st.markdown("---")
    
    # ➕ New Chat Button
    if st.button("➕ Haasaa Haarawaa (New Chat)", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("📜 Seenaa Haasaa (Recents)")

    # Load chats specific to logged in user
    if supabase and st.session_state.user:
        try:
            response = supabase.table("chat_logs") \
                .select("session_id, chat_title, created_at") \
                .eq("user_id", st.session_state.user.id) \
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
# Main Chat Application Window
# ---------------------------------------------------------
st.subheader(f"💬 {st.session_state.current_chat_title}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    if not st.session_state.messages or st.session_state.current_chat_title == "Haasaa Haarawaa":
        st.session_state.current_chat_title = prompt[:30]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [
        {"role": "system", "content": "Ati Orochat-ai dha. Mohammed Kalif Umar tu si uume. Deebii kee Afaan Oromoo qulqulluudhaan deebisi."},
        {"role": "user", "content": prompt}
    ]

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=payload,
                    temperature=0.6,
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
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
