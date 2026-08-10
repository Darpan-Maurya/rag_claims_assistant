import requests
import streamlit as st
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("RAG_API_KEY")

st.set_page_config(
    page_title="RAG-Powered Insurance Claims Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("Insurance Claims Query Assistant")
st.caption("Ask natural language questions on insurance claims data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


def headers():
    result = {}
    if API_KEY:
        result["X-API-Key"] = API_KEY
    return result


def post_feedback(rating: str, notes: str = ""):
    last = st.session_state.last_response
    if not last:
        return
    response = requests.post(
        f"{API_BASE_URL}/feedback",
        json={
            "request_id": last["request_id"],
            "conversation_id": last.get("conversation_id"),
            "rating": rating,
            "notes": notes or None,
        },
        headers=headers(),
        timeout=20,
    )
    response.raise_for_status()

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
                f"{API_BASE_URL}/query",
                json={
                    "query": user_query,
                    "top_k": 25,
                    "conversation_id": st.session_state.last_response.get("conversation_id")
                    if st.session_state.last_response else None,
                },
                headers=headers(),
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["answer"]
            st.session_state.last_response = payload

    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text
        if status_code == 503:
            answer = (
                "The requested data service is not ready yet. Analytics and LLM-only questions can run "
                "without Qdrant; RAG questions need a completed ingestion job. "
                + (detail or "Check the backend `/ready` endpoint.")
            )
        else:
            answer = detail or f"Backend request failed with status {status_code}."
    except requests.RequestException:
        answer = "The backend could not be reached. Check the API URL and service health."

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)

if st.session_state.last_response:
    response = st.session_state.last_response
    left, right = st.columns([2, 1])
    with left:
        if response.get("evidence"):
            st.subheader("Evidence")
            st.dataframe(response["evidence"], use_container_width=True, hide_index=True)
    with right:
        st.subheader("Request")
        st.write(f"Route: `{response.get('route')}`")
        st.write(f"Request ID: `{response.get('request_id')}`")
        if response.get("warnings"):
            st.warning("\n".join(response["warnings"]))

        notes = st.text_area("Feedback notes", key="feedback_notes", height=90)
        col_up, col_down, col_neutral = st.columns(3)
        try:
            if col_up.button("Good"):
                post_feedback("up", notes)
                st.success("Feedback stored.")
            if col_down.button("Bad"):
                post_feedback("down", notes)
                st.success("Feedback stored.")
            if col_neutral.button("Neutral"):
                post_feedback("neutral", notes)
                st.success("Feedback stored.")
        except Exception as e:
            st.error(f"Feedback failed: {e}")

    with st.expander("Retrieval details"):
        st.json(
            {
                "filters": response.get("filters", {}),
                "retrieval_summary": response.get("retrieval_summary", {}),
                "conversation_id": response.get("conversation_id"),
            }
        )
