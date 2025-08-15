import os
import streamlit as st

st.set_page_config(
    page_title="Agent Lambda",
    page_icon="🧠",
    layout="wide",
)
cwd = os.getcwd()
main_page = st.Page(page=os.path.join(cwd, "app.py"), title="Home", icon="🏠")
feedback_page = st.Page(page=os.path.join(cwd, "utils", "feedback.py"), title="Feedbacks", icon="🗒️")
account_page = st.Page(page=os.path.join(cwd, "utils", "account.py"), title="Account", icon="⚙️")

if st.user.is_logged_in:
    pages = [main_page, feedback_page, account_page]
else:
    pages = [main_page, feedback_page]

pg = st.navigation(pages=pages, position="top")
pg.run()
