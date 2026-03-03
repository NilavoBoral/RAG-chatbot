import streamlit as st
from rag import ask, new_session, _load_all_sessions

st.set_page_config(page_title="PDF Assistant", page_icon="📄")

st.title("📄 PDF Assistant")

with st.sidebar:
    st.header("💬 Chats")

    if st.button("➕ New Chat"):
        st.session_state.session_id = new_session()
        st.rerun()
        
# Create backend session once
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session()

sessions = _load_all_sessions()
history = sessions.get(st.session_state.session_id, [])

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask your question...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(prompt, st.session_state.session_id)

            st.markdown(result["answer"])

            # if result["sources"]:
            #     st.caption("Sources:")
            #     for s in result["sources"]:
            #         st.caption(f"- {s}")

    # st.rerun()