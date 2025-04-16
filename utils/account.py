import os
import streamlit as st
from utils.database import delete_all_sessions_for_user
from utils.database import get_unique_users_and_session_counts
import pandas as pd

st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="{st.experimental_user.picture}" width="40" height="40" style="border-radius: 50%; margin-right: 10px; vertical-align: middle;">
        <span style="font-size: 1.1em;">{st.experimental_user.name}</span>
    </div>
    """,
    unsafe_allow_html=True
)
# Logout
with st.container(border=True):
    st.markdown("<br>", unsafe_allow_html=True) # Add a little space inside top border
    google_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-top: 20px; margin-bottom: 20px;">
            <img src="{google_logo_url}" width="25" height="25" style="margin-right: 10px; vertical-align: middle;">
            <!-- The button itself will be rendered below by Streamlit -->
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        if st.button("Confirm Logout", use_container_width=True):
            st.logout()
            
    st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Delete All Chat Sessions"):
    if st.button("Clear All Chats"):
        deleted_count = delete_all_sessions_for_user(st.session_state.email)
        st.success(f"Deleted {deleted_count} session(s)")
        
st.divider()
st.subheader("Admin Section")
if st.experimental_user.is_logged_in:
    if st.experimental_user.email == os.environ.get("ADMIN_EMAIL"):
        users = get_unique_users_and_session_counts()
        if users:
            with st.spinner("Loading unique users and session counts..."):
                st.metric(
                    label="Total Unique Users",
                    value=len(users),
                    delta=None,
                    help="Total number of unique users in the system.",
                    border=True          
                    )
                st.divider()
                df = pd.DataFrame(users)
                df.columns = ["User ID", "Session Count"]
                st.dataframe(df, use_container_width=True)