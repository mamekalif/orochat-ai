import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Afaan Oromoo AI", page_icon="🇪🇹", layout="centered")
st.title("🇪🇹 Afaan Oromoo AI (Small)")
st.caption("Fayyadamaaf qophaa'e – permanent free demo")

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Stronger system prompt – forces more pure Afaan Oromoo
SYSTEM = """Ati gargaaraa Afaan Oromoo qofa ti.
Deebii kee hunda Afaan Oromoo (Oromiffaa) qulqulluu ta'ee kenni.
Afaan Soomaali, Afaan Amaaraa yookiin English hin fayyadamiin.
Yoo gaaffiin Afaan Oromoo ta'e, deebii keesaa Afaan Oromoo qofa ta'uu qaba.
Fakkeenya deebii gaarii:
- "Nagaa, galata Waaqayyo. Ati akkam jirta?"
- "Maqaan koo Grok. Ati maqaa kee natti himuu dandeessa?"
Yoo hin beekne, "Hin beeku" jedhi.
Deebii kee gabaabaa fi ifaa ta'uu qaba."""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Gaaffii kee barreessi..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation
    messages = [{"role": "system", "content": SYSTEM}]
    for m in st.session_state.messages[-6:]:  # keep last few turns
        messages.append({"role": m["role"], "content": m["content"]})

    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.65,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            st.markdown(response)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})
