
import streamlit as st
import hmac
from config import DASHBOARD_PASSWORD


def show_login():
    if st.session_state.get("auth", False):
        return True

    st.title("Digital Eye Login")
    st.write("Please enter your dashboard password.")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if hmac.compare_digest(password, DASHBOARD_PASSWORD):
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return st.session_state.get("auth", False)
