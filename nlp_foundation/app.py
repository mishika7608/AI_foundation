from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from groq import Groq
# from streamlit_js_eval import streamlit_js_eval

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Interview Chatbot", page_icon="💬")
st.title("Interview Chatbot")

# -------------------------
# Session state
# -------------------------
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False
if "chat_complete" not in st.session_state:
    st.session_state.chat_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Helpers
# -------------------------
def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True

# -------------------------
# Setup phase
# -------------------------
if not st.session_state.setup_complete:
    st.subheader("Personal Information")

    st.session_state.setdefault("name", "")
    st.session_state.setdefault("experience", "")
    st.session_state.setdefault("skills", "")

    st.session_state["name"] = st.text_input("Name", st.session_state["name"])
    st.session_state["experience"] = st.text_area("Experience", st.session_state["experience"])
    st.session_state["skills"] = st.text_area("Skills", st.session_state["skills"])

    st.subheader("Company and Position")

    st.session_state.setdefault("level", "Junior")
    st.session_state.setdefault("position", "Data Scientist")
    st.session_state.setdefault("company", "Amazon")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state["level"] = st.radio(
            "Choose level",
            ["Junior", "Mid-level", "Senior"],
            index=["Junior", "Mid-level", "Senior"].index(st.session_state["level"])
        )

    with col2:
        st.session_state["position"] = st.selectbox(
            "Choose a position",
            ["Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"],
            index=["Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"].index(st.session_state["position"])
        )

    st.session_state["company"] = st.selectbox(
        "Select a Company",
        ["Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"],
        index=["Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"].index(st.session_state["company"])
    )

    if st.button("Start Interview", on_click=complete_setup):
        st.success("Setup complete. Starting interview...")

# -------------------------
# Interview phase
# -------------------------
if st.session_state.setup_complete and not st.session_state.feedback_shown and not st.session_state.chat_complete:

    st.info("Start by introducing yourself 👋")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env")
        st.stop()
    client = Groq(api_key=api_key)

    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "system",
            "content": (
                f"You are an HR executive that interviews an interviewee called {st.session_state['name']}. "
                f"Experience: {st.session_state['experience']}. "
                f"Skills: {st.session_state['skills']}. "
                f"Position: {st.session_state['level']} {st.session_state['position']} "
                f"at {st.session_state['company']}."
            )
        }]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.user_message_count < 5:
        if prompt := st.chat_input("Your response"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.user_message_count += 1

            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.user_message_count < 5:
                with st.chat_message("assistant"):
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=st.session_state.messages
                    )
                    reply = completion.choices[0].message.content
                    st.markdown(reply)

                st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.user_message_count >= 5:
        st.session_state.chat_complete = True

# -------------------------
# Feedback button
# -------------------------
if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button("Get Feedback", on_click=show_feedback):
        st.info("Generating feedback...")

# -------------------------
# Feedback phase
# -------------------------
if st.session_state.feedback_shown:
    st.subheader("Feedback")

    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env")
        st.stop()
    feedback_client = Groq(api_key=groq_api_key)

    feedback = feedback_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content":"""
                    You are a helpful tool that provides feedback on an interviewee performance.
                    Before the Feedback give a score of 1 to 10.
                    Follow this format:
                    Overal Score: //Your score
                    Feedback: //Here you put your feedback
                    Give only the feedback do not ask any additional questins.
                """
            },
            {"role": "user", "content": f"This is the interview you need to evaluate. Keep in mind that you are only a tool. And you shouldn't engage in any converstation: {conversation}"}
        ]
    )

    st.write(feedback.choices[0].message.content)

    def reset_app():
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    if st.button("Restart Interview", type="primary"):
        reset_app()
    # if st.button("Restart Interview", type="primary"):
    #     streamlit_js_eval(js_expressions="parent.window.location.reload()")
