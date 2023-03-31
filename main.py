"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage


@st.cache_resource
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        # openai_api_key=None,
        max_tokens=1024,
        temperature=0,
    )
    chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
    )
    return chain

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# Initialize or reload Chain
if "chain" not in st.session_state:
    chain = load_chain()
    st.session_state['chain'] = chain

with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_area("Send a message...")
    submitted = st.form_submit_button()

    if submitted:
        st.session_state.chain.run(input=user_input)

# Display chat history
try:
    messages = st.session_state.chain.memory.chat_memory.messages
    for i, msg in list(enumerate(messages))[::-1]:
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=f"{i}_user", avatar_style="thumbs")
        else:
            message(msg.content, key=f"{i}")
except KeyError:
    pass
