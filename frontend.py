"""Streamlit dev UI.

Kept for local development and demos. The deployed UI is the static page in ``web/`` -
Streamlit needs a long-lived websocket server, which Vercel does not provide.

Run with:  uv run --extra local streamlit run frontend.py
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.environ.get("SAFESPACE_BACKEND", "http://localhost:8000")
REQUEST_TIMEOUT = 90

st.set_page_config(page_title="SafeSpace", page_icon="💬", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    # Stable per browser session, which is what gives the backend conversation memory.
    st.session_state.session_id = uuid.uuid4().hex

st.title("SafeSpace")
st.caption("A place to think out loud.")

with st.sidebar:
    st.subheader("Settings")
    region = st.selectbox(
        "Your region",
        options=["INTL", "US", "UK", "IN", "CA", "AU"],
        help="Used to show the right crisis helplines.",
    )
    if st.button("Start a new conversation"):
        st.session_state.chat_history = []
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

    st.divider()
    st.subheader("Need someone now?")
    try:
        resources = requests.get(
            f"{BACKEND_URL}/crisis-resources", params={"region": region}, timeout=10
        ).json()
        for line in resources.get("lines", []):
            st.markdown(f"**{line['name']}**  \n{line['contact']}")
    except requests.RequestException:
        st.markdown("**Find a helpline:** https://findahelpline.com")

st.info(
    "SafeSpace is an AI support tool, not a therapist. It does not provide medical "
    "advice, diagnosis, or treatment. In an emergency, contact your local emergency "
    "services.",
    icon="ℹ️",
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool"):
            st.caption(f"via {message['tool']}")

user_input = st.chat_input("What's on your mind today?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Every failure mode below used to surface as a raw traceback in the UI:
            # no status check, no timeout, and a bare ["response"] lookup.
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "message": user_input,
                        "session_id": st.session_state.session_id,
                        "region": region,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.Timeout:
                reply, tool = "That took too long to come back. Could you try again?", None
            except requests.RequestException:
                reply, tool = (
                    f"I can't reach the SafeSpace backend at {BACKEND_URL}. "
                    "Is it running? Start it with `uv run python -m backend.main`.",
                    None,
                )
            else:
                if response.status_code == 429:
                    reply, tool = "That's a lot of messages at once - give me a moment.", None
                elif response.status_code >= 400:
                    body = response.json() if response.content else {}
                    reply = body.get("detail", "Something went wrong on my end. Please try again.")
                    tool = None
                else:
                    body = response.json()
                    reply = body.get("response") or "I didn't catch that - could you say it again?"
                    tool = body.get("tool_called")
                    if body.get("session_id"):
                        st.session_state.session_id = body["session_id"]

        st.markdown(reply)
        if tool:
            st.caption(f"via {tool}")

    st.session_state.chat_history.append({"role": "assistant", "content": reply, "tool": tool})
