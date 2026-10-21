import streamlit as st
from groq import Groq

# Page layout setup
st.set_page_config(
    page_title="Orochat-ai 🇪🇹", 
    page_icon="🇪🇹", 
    layout="centered"
)

st.title("🇪🇹 Orochat-ai")
st.caption("Gargaaraa Afaan Oromoo – Powered by Groq")

# Initialize client using Streamlit secrets
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# Dynamic Model Selector: Prevents breaks from model deprecations
@st.cache_data(ttl=3600)  # Caches active model list for 1 hour
def get_active_groq_model():
    """Dynamically fetches active text models from Groq API to avoid 400/404 errors."""
    try:
        available_models = client.models.list()
        model_ids = [m.id for m in available_models.data]
        
        # Priority list of production text models
        priority_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
        
        # Return the first matching active priority model
        for model in priority_models:
            if model in model_ids:
                return model
                
        # Fallback: Pick any non-whisper/non-guard active text model
        for m_id in model_ids:
            if not any(x in m_id.lower() for x in ["whisper", "guard", "vision", "audio"]):
                return m_id
                
        return "llama-3.1-8b-instant"
    except Exception:
        return "llama-3.1-8b-instant"

ACTIVE_MODEL = get_active_groq_model()

# High-precision Afaan Oromoo system instructions with exact identity definition
SYSTEM_PROMPT = """Ati gargaaraa Afaan Oromoo intelliijenti ta'ee dha. 
Maqaan kee 'Orochat-ai' jedhama. 
Ati gargaaraa teeknoolojii AI fi afaanii ti.

Seera Deebii:
1. Deebii kee hunda Afaan Oromoo qulqulluu, bareedaa fi sirrii ta'een deebisi.
2. Gaaffii maqaa ykn eenyummaa kee ilaalchisee: "Maqaan koo Orochat-ai ti, gargaaraa Afaan Oromoo fi teeknoolojii ti" jedhi.
3. Tasumaa "ChatGPT", "OpenAI" yookiin maqaa qaama biroo jedhii hin waamin.
4. Deebii kee gabaabaa, ifaa fi kabaja qabu ta'uu qaba.

Fakkeenya:
Gaaffii: Maqaan kee eenyu?
Deebii: Maqaan koo Orochat-ai ti. Akkamittin si gargaaruu danda'a?

Gaaffii: Akkam jirtu?
Deebii: Fayyaa dha, galatoomaa! Isin akkam jirtu?
"""

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    # Append user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context payload with system prompt + last 6 turns
    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    # Stream response to UI
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
                
                # Store assistant response to memory
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
