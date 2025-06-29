import streamlit as st
import random
import time
from GBackend import rag_query


# --------------------------------------------------
# 🖼️ Page configuration & header
# --------------------------------------------------
st.set_page_config(
    page_title="Jupiter FAQ Chatbot",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <h1 style='text-align: center;'>🔷 Jupiter Money FAQ Chatbot</h1>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 💾 Session state (conversation memory)
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []

# 🔹 Helpful tips and dynamic statuses
TIPS = [
    "💡 Tip: Ask \"How does Jupiter categorise my spending?\"",
    "💡 Tip: Try \"What can I do in the Money tab?\"",
    "💡 Tip: Wondering about rewards? Ask \"Tell me about card rewards.\"",
    "💡 Tip: Curious about KYC? Try \"What documents are required for KYC?\"",
]

STATUSES = [
    "🔍 Searching Jupiter knowledge base…",
    "📚 Reading related FAQs…",
    "⚙️ Crunching the numbers…",
    "💡 Generating insights…",
    "✏️ Drafting a clear answer…",
    "🔄 Double‑checking details…",
]


# --------------------------------------------------
# 🗨️ Show conversation history
# --------------------------------------------------
st.markdown("## ")  # Spacer below header
chat_placeholder = st.container()
with chat_placeholder:
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)


# --------------------------------------------------
# ✍️ Chat input
# --------------------------------------------------
user_query = st.chat_input("Ask me anything about Jupiter…")


# --------------------------------------------------
# 🤖 Generate & stream assistant reply
# --------------------------------------------------
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        status_sequence = ["🤔 Thinking…"]
        status_sequence += random.sample(STATUSES, k=3)
        status_sequence += random.sample(TIPS, k=random.randint(1, 2))
        random.shuffle(status_sequence[1:])  # keep "Thinking…" first

        for status in status_sequence:
            message_placeholder.markdown(status)
            time.sleep(random.uniform(0.5, 1.0))

        response = rag_query(user_query)

        full_response = ""
        for word in response.split():
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.04)

        message_placeholder.markdown(full_response)

    st.session_state.history.append((user_query, full_response))
