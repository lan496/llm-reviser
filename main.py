"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import pyperclip

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder


ENGLISH_REVISER_PROMPT = """You are a helpful assistant for correcting English spelling and improving sentences.
Replace immature words and sentences with more beautiful and elegant ones.
Keep the same meaning, but make it more literary.
"""

@st.cache_resource
def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        # openai_api_key=None,
        max_tokens=4096,
        temperature=0,
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ENGLISH_REVISER_PROMPT),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="history"),
    ])

    chain = ConversationChain(
        llm=llm,
        prompt=chat_prompt,
        # Somehow, return_messages=True is required...
        # Ref: https://github.com/hwchase17/langchain/issues/1971
        memory=ConversationBufferMemory(return_messages=True),
        verbose=True,
    )
    return chain


def main():
    # From here down is all the StreamLit UI.
    st.set_page_config(page_title="ChatGPT for Poor Man", page_icon=":robot:")
    st.header("ChatGPT for Poor Man")

    # Show prompt
    st.info(f"Prompt: {ENGLISH_REVISER_PROMPT}")

    # Initialize or reload Chain
    if "chain" not in st.session_state:
        chain = load_chain()
        st.session_state['chain'] = chain

    # Prepare input
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_area("Send a message...")
        submitted = st.form_submit_button()

        if submitted and (user_input != ""):
            with st.spinner("Wait for AI..."):
                st.session_state.chain.predict(input=user_input)

    # Display chat history
    try:
        messages = st.session_state.chain.memory.chat_memory.messages
        for i, msg in list(enumerate(messages))[::-1]:
            if isinstance(msg, HumanMessage):
                message(msg.content, is_user=True, key=f"{i}_user", avatar_style="thumbs")
            else:
                col_msg, col_button = st.columns([9, 1])
                with col_msg:
                    message(msg.content, key=f"{i}")
                with col_button:
                    if st.button('Copy', key=f'{i}_copy'):
                        pyperclip.copy(msg.content)
    except KeyError:
        pass


if __name__ == '__main__':
    main()
