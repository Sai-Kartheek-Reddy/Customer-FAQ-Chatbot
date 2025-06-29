import subprocess
import sys
import random
import time

# --------------------------------------------------
# 🔧 Install / upgrade dependencies silently
# --------------------------------------------------
# subprocess.check_call([
#     sys.executable,
#     "-m",
#     "pip",
#     "install",
#     "-q",
#     "-U",
#     "trl",
#     "faiss-cpu",
#     "langchain",
#     "sentence-transformers",
#     "langchain-community",
#     "streamlit",
# ])


import streamlit as st
from Backend import rag_query  # Replace "Backend" with "GBackend" to use Gemini Based Applications

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

# 🔹 Helpful tips we can show *during* generation
TIPS = [
    "💡 Tip: Ask \"How does Jupiter categorise my spending?\"",
    "💡 Tip: Try \"What can I do in the Money tab?\"",
    "💡 Tip: Wondering about rewards? Ask \"Tell me about card rewards.\"",
    "💡 Tip: Curious about KYC? Try \"What documents are required for KYC?\"",
]

# 🔹 Dynamic status phrases
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
# ✍️ Chat‑input (auto‑bottom)
# --------------------------------------------------
user_query: str | None = st.chat_input("Ask me anything about Jupiter…")

# --------------------------------------------------
# 🤖 Generate & *stream* assistant reply
# --------------------------------------------------
if user_query:
    # ➊ Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # ➋ Prepare assistant message placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # ➌ Build a dynamic sequence of statuses + tips
        status_sequence: list[str] = ["🤔 Thinking…"]
        status_sequence += random.sample(STATUSES, k=3)
        status_sequence += random.sample(TIPS, k=random.randint(1, 2))
        random.shuffle(status_sequence[1:])  # shuffle everything except the very first "Thinking…"

        for status in status_sequence:
            message_placeholder.markdown(status)
            time.sleep(random.uniform(0.5, 1.0))

        # ➍ Call backend RAG
        response = rag_query(user_query)

        # ➎ Live‑typing effect (word‑by‑word)
        full_response = ""
        for word in response.split():
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.04)

        # ➏ Replace cursor ▌ with final text
        message_placeholder.markdown(full_response)

    # ➐ Persist interaction into session history
    st.session_state.history.append((user_query, full_response))
