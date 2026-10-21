import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Afaan Oromoo AI", 
    page_icon="🇪🇹", 
    layout="centered"
)

st.title("🇪🇹 Afaan Oromoo AI")
st.caption("Fayyadamaaf qophaa'e – Powered by Groq")

# Initialize Groq client securely
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# System prompt for high-quality Afaan Oromoo
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

# User Chat Input
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        payload.append({"role": m["role"], "content": m["content"]})

    # List of candidate models to try in sequence
    MODELS_TO_TRY = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            stream = None
            last_err = None

            # Attempt each model until one succeeds
            for model_id in MODELS_TO_TRY:
                try:
                    stream = client.chat.completions.create(
                        model=model_id,
                        messages=payload,
                        temperature=0.5,
                        max_tokens=300,
                        stream=True
                    )
                    break  # Success! Break loop
                except Exception as err:
                    last_err = err
                    continue

            if stream is not None:
                reply = st.write_stream(
                    chunk.choices[0].delta.content or "" for chunk in stream
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                st.error(f"Dogoggorri uumamee jira: {str(last_err)}")
                
