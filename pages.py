import os
import streamlit as st
    
cwd = os.getcwd()
main_page = st.Page(page=os.path.join(cwd, "app.py"), title="Home", icon="🎓")
feedback_page = st.Page(page=os.path.join(cwd, "utils", "feedback.py"), title="Feedbacks", icon="🗒️")
account_page = st.Page(page=os.path.join(cwd, "utils", "account.py"), title="Account", icon="⚙️")

if st.experimental_user.is_logged_in:
    pages = [main_page, feedback_page, account_page]
else:
    pages = [main_page, feedback_page]

pg = st.navigation(pages=pages)
pg.run()