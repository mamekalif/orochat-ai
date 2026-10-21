import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Afaan Oromoo AI", page_icon="🇪🇹", layout="centered")
st.title("🇪🇹 Afaan Oromoo AI")
st.caption("Powered by Gemini 1.5 Flash")

# Configure Gemini
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

SYSTEM_PROMPT = """Ati gargaaraa Afaan Oromoo ti.
Deebii kee hunda Afaan Oromoo qulqulluu fi gabaabaa ta'ee deebisi.
English yookiin Afaan biraa fayyadamuu hin danda'atamu.

Fakkeenya:
Gaaffii: Akkam jirtu?
Deebii: Fayyaa dha, galatoomaa! Isin akkam jirtu?
"""

# Initialize model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Gaaffii kee Afaan Oromootiin barreessi..."):
    # Store and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history for Gemini chat API
    history = []
    for m in st.session_state.messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=history)

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            try:
                response = chat.send_message(prompt)
                reply = response.text
                st.markdown(reply)
                
                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Dogoggorri uumamee jira: {str(e)}")
                
