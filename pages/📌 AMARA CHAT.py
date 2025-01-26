import streamlit as st
from utils import apply_styles
from agents import agent, as_stream

st.title("AMARA CHAT")

if st.button("💬 New Chat"):
  st.session_state.messagesA = []
  st.rerun()

apply_styles()

if "messagesA" not in st.session_state:
  st.session_state.messagesA = []

for message in st.session_state.messagesA:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
  st.session_state.messagesA.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    chunks = agent.run(prompt, stream=True)
    response = st.write_stream(as_stream(chunks))
  st.session_state.messagesA.append({"role": "assistant", "content": response})