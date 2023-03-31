import os

import streamlit as st
from streamlit_chat import message
import pyperclip
from openai.error import AuthenticationError

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
def load_chain(openai_api_key: str):
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        openai_api_key=openai_api_key,
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
    title = "English Reviser for Poor Man"
    st.set_page_config(page_title=title, page_icon=":robot:")
    st.header(title)

    # API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        if "openai_api_key" not in st.session_state:
            with st.form("api-key-form", clear_on_submit=True):
                openai_api_key = st.text_area('Your OpenAI API key')
                submitted = st.form_submit_button('Save')
                if submitted and (openai_api_key != ""):
                    st.session_state['openai_api_key'] = openai_api_key
                    st.experimental_rerun()
                else:
                    return  # Reload page
        else:
            openai_api_key = st.session_state.openai_api_key

    # Show prompt
    st.info(f"Prompt: {ENGLISH_REVISER_PROMPT}")

    # Initialize or reload Chain
    if "chain" not in st.session_state:
        chain = load_chain(openai_api_key)
        st.session_state['chain'] = chain

    # Prepare input
    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_area("Send a message...")
        submitted = st.form_submit_button()

        if submitted and (user_input != ""):
            with st.spinner("Wait for AI..."):
                try:
                    st.session_state.chain.predict(input=user_input)
                except AuthenticationError:
                    st.error("Failed to authenticate OpenAI API key. Please reload page and specify a correct API key.")
                    st.session_state.pop('openai_api_key')
                    st.experimental_rerun()

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
