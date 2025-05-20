import streamlit as st
from main import answer_question

st.set_page_config(page_title="LEOparts Chatbot", page_icon="⚙️")

st.title("⚙️  LEOparts Standards & Attributes Chatbot")

with st.form(key="qa_form"):
    user_q = st.text_input(
        "Ask a question about parts, colours, cavities…",
        placeholder="e.g. Any white parts?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and user_q.strip():
    with st.spinner("Thinking…"):
        reply = answer_question(user_q)
    st.markdown("#### Answer")
    st.write(reply)
