import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Afaan Oromoo AI", page_icon="🇪🇹")
st.title("Afaan Oromoo AI (Small)")
st.write("Fayyadamaaf qophaa'e – permanent free demo")

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

SYSTEM = """You are a helpful assistant that speaks Afaan Oromoo.
Always reply in Afaan Oromoo when the user writes in Afaan Oromoo.
Keep answers short and natural."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Gaaffii kee barreessi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": SYSTEM}]
    for m in st.session_state.messages[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with st.spinner("Yaadaa jira..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            st.markdown(response.strip())

    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
