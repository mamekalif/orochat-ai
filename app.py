import streamlit as st
from groq import Groq

# Page layout setup
st.set_page_config(page_title="Afaan Oromoo AI", page_icon="🇪🇹", layout="centered")
st.title("🇪🇹 Afaan Oromoo AI")
st.caption("Fayyadamaaf qophaa'e – Powered by Groq")

# Initialize client using Streamlit secrets
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# Dynamic Model Selector: Ensures the app never breaks due to model deprecation
@st.cache_data(ttl=3600)  # Caches active model for 1 hour
def get_active_groq_model():
    """Dynamically fetches active text models from Groq API to avoid 400/404 errors."""
    try:
        available_models = client.models.list()
        model_ids = [m.id for m in available_models.data]
        
        # Priority list of top production models
        priority_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
        
        # Return the first matching priority model active on your account
        for model in priority_models:
            if model in model_ids:
                return model
                
        # Fallback: Pick any non-whisper/non-guard active model
        for m_id in model_ids:
            if not any(x in m_id.lower() for x in ["whisper", "guard", "vision", "audio"]):
                return m_id
                
        return "llama-3.1-8b-instant"
    except Exception:
        # Emergency default fallback
        return "llama-3.1-8b-instant"

# Select the working model dynamically
ACTIVE_MODEL = get_active_groq_model()

# High-quality Afaan Oromoo system instructions
SYSTEM_PROMPT = """Ati gargaaraa Afaan Oromoo ti.
Deebii kee hunda Afaan Oromoo qulqulluu, sirrii fi gabaabaa ta'ee deebisi.
English yookiin Afaan biraa fayyadamuu hin danda'atamu.

Fakkeenya:
Gaaffii: Akkam jirtu?
Deebii: Fayyaa dha, galatoomaa! Isin akkam jirtu?

Gaaffii: Magaalaan guddoon Itoophiyaa eessa?
Deebii: Magaalaan guddoon Itoophiyaa Finfinnee dha.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input handler
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model=ACTIVE_MODEL,
                    messages=payload,
                    temperature=0.5,
                    max_tokens=350,
                    stream=True
                )

                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )
                
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
