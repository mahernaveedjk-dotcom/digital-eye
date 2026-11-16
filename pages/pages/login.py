import hmac
import streamlit as st

from config import DASHBOARD_PASSWORD


def show_login():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        return True

    st.title("Digital Eye")
    st.caption("Security dashboard - please log in")

    with st.form("login"):
        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button("Log in")

    if submitted:
        if hmac.compare_digest(
            password.encode(),
            DASHBOARD_PASSWORD.encode()
        ):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Wrong password")

    return False
