import streamlit as st
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="Afaan Oromoo AI", 
    page_icon="🇪🇹", 
    layout="centered"
)

st.title("🇪🇹 Afaan Oromoo AI")
st.caption("Fayyadamaaf qophaa'e – Powered by Groq & Llama 3.3")

# Initialize Groq client securely from Streamlit secrets
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# High-precision system instructions with Afaan Oromoo few-shot guidance
SYSTEM_PROMPT = """Ati gargaaraa Afaan Oromoo ti.
Deebii kee hunda Afaan Oromoo qulqulluu, sirrii fi gabaabaa ta'ee deebisi.
English yookiin Afaan biraa fayyadamuu hin danda'atamu.

Fakkeenya:
Gaaffii: Akkam jirtu?
Deebii: Fayyaa dha, galatoomaa! Isin akkam jirtu?

Gaaffii: Magaalaan guddoon Itoophiyaa eessa?
Deebii: Magaalaan guddoon Itoophiyaa Finfinnee dha.
"""

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    # Append user prompt to UI history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context payload with system prompt + last 6 turns
    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    # Generate streaming assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=payload,
                    temperature=0.5,
                    max_tokens=300,
                    stream=True
                )

                # Direct stream response output to Streamlit UI
                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )

                # Store assistant response to memory
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
