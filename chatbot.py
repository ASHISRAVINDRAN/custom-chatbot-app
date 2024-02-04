import streamlit as st
from chatgpt import MITKOpenAI
# App title
st.set_page_config(page_title="💬 MITK buddy")

# Sidebar
with st.sidebar:
    st.title('💬 MITK buddy')
    st.selectbox('Supported Models:', ('OpenAI', 'Mistral'))
    is_rag: bool = st.checkbox('Retrieval Augmented Generation')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if "model" not in st.session_state.keys():
    st.session_state.model = MITKOpenAI()
# Function for generating LLM response
def generate_response(prompt_input):
    return st.session_state.model.get_response(prompt_input, is_rag)

# # User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)