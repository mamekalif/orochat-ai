import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Afaan Oromoo AI", page_icon="🇪🇹", layout="centered")
st.title("🇪🇹 Afaan Oromoo AI")
st.caption("Powered by Gemini 2.5 Flash")

# Initialize Gemini Client using Streamlit Secrets
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_gemini_client()

SYSTEM_PROMPT = """Ati gargaaraa Afaan Oromoo ti.
Deebii kee hunda Afaan Oromoo qulqulluu fi gabaabaa ta'ee deebisi.
English yookiin Afaan biraa fayyadamuu hin danda'atamu.

Fakkeenya:
Gaaffii: Akkam jirtu?
Deebii: Fayyaa dha, galatoomaa! Isin akkam jirtu?
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history for API request
    contents = []
    for m in st.session_state.messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                # Stream the output directly
                response = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.6,
                    )
                )

                reply = st.write_stream(chunk.text for chunk in response)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
