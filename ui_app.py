import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="RAG-Powered Insurance Claims Assistant",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Insurance Claims Query Assistant")
st.caption("Ask natural language questions on insurance claims data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask a question about claims data...")

if user_query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call FastAPI backend
    try:
        with st.spinner("Analyzing claims..."):
            response = requests.post(
                API_URL,
                json={"query": user_query, "top_k": 25},
                timeout=60,
            )
            response.raise_for_status()
            answer = response.json()["answer"]

    except Exception as e:
        answer = f"❌ Error communicating with backend: {e}"

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
