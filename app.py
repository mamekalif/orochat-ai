import streamlit as st
from groq import Groq
import json
from datetime import datetime
import os

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Orochat-ai 🇪🇹", 
    page_icon="🇪🇹", 
    layout="centered"
)

st.title("🇪🇹 Orochat-ai")
st.caption("Gargaaraa Afaan Oromoo Smart AI – Powered by Groq & RAG Engine")

# ---------------------------------------------------------
# API Client & Dynamic Model Loader
# ---------------------------------------------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

@st.cache_data(ttl=3600)
def get_active_groq_model():
    """Dynamically fetches active text models from Groq API."""
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
# Layer 1: Long-Term Logging Function (Data Gathering)
# ---------------------------------------------------------
LOG_FILE = "chat_logs.jsonl"

def log_conversation(user_prompt, ai_response, feedback=None):
    """Saves every interaction for future fine-tuning and model learning."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_prompt,
        "assistant": ai_response,
        "feedback": feedback
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------
# Layer 3: Dynamic Knowledge Base (RAG Engine)
# ---------------------------------------------------------
# Easily expand this dictionary with proverbs, definitions, or custom domain knowledge
KNOWLEDGE_BASE = {
    "mammaaksa": "Mammaaksa: 'Haadha manaa fi abbaa manaa walmormu, alagaatu gidduutti fayyadama.'",
    "baabbile": "Magaalaan Baabbilee Godina Bahaa Harargee keessatti argamti. Naannoo bosona leencotaa fi Arbaatiin beekamti.",
    "orochat": "Orochat-ai gargaaraa teeknoolojii AI fi Afaan Oromoo kan Mohammed Kalif Umar tu ijaareedha.",
    "mohammed": "Mohammed Kalif Umar kalaqaa fi ijaaraa Orochat-ai fi teeknoolojii Afaan Oromoo ti."
}

def retrieve_knowledge(user_query):
    """Searches local knowledge base and injects relevant context into the prompt."""
    query_lower = user_query.lower()
    retrieved_facts = []
    
    for key, fact in KNOWLEDGE_BASE.items():
        if key in query_lower:
            retrieved_facts.append(fact)
            
    if retrieved_facts:
        return "\n".join(retrieved_facts)
    return None

# ---------------------------------------------------------
# Core System Prompt
# ---------------------------------------------------------
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
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = None

# ---------------------------------------------------------
# UI: Display History & Feedback Buttons
# ---------------------------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Layer 2: Feedback Interface
if st.session_state.last_interaction:
    st.markdown("---")
    st.caption("Deebii kana irratti yaada kee kenni (Help Orochat-ai learn):")
    col1, col2, _ = st.columns([1, 1, 4])
    
    with col1:
        if st.button("👍 Deebii Gaarii", key="fb_good"):
            log_conversation(
                st.session_state.last_interaction["user"],
                st.session_state.last_interaction["assistant"],
                feedback="positive"
            )
            st.toast("Galatoomaa! Yaadni keessan galmeeffameera.")
            st.session_state.last_interaction = None

    with col2:
        if st.button("👎 Deebii Dogoggoraa", key="fb_bad"):
            log_conversation(
                st.session_state.last_interaction["user"],
                st.session_state.last_interaction["assistant"],
                feedback="negative"
            )
            st.toast("Galatoomaa! Sirreessuuf data keessatti galmeessineerra.")
            st.session_state.last_interaction = None

# ---------------------------------------------------------
# Chat Input & Inference Pipeline
# ---------------------------------------------------------
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build base payload
    payload = [{"role": "system", "content": SYSTEM_PROMPT}]

    # RAG Step: Retrieve relevant dynamic context if match found
    context = retrieve_knowledge(prompt)
    if context:
        payload.append({
            "role": "system", 
            "content": f"Odeeffannoo Dabalataa (Knowledge Base Context):\n{context}"
        })

    # Add conversation memory context (last 6 turns)
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=payload,
                    temperature=0.6,
                    max_tokens=350,
                    stream=True
                )

                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )

                # Store response in session memory
                st.session_state.messages.append({"role": "assistant", "content": reply})

                # Set for feedback & log interaction immediately
                st.session_state.last_interaction = {
                    "user": prompt,
                    "assistant": reply
                }
                log_conversation(prompt, reply)
                st.rerun()

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
