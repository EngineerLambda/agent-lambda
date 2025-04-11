import os
import streamlit as st
    
cwd = os.getcwd()
main_page = st.Page(page=os.path.join(cwd, "app.py"), title="Home", icon="🎓")
feedback_page = st.Page(page=os.path.join(cwd, "feedback.py"), title="Feedbacks", icon="🗒️")

pg = st.navigation(pages=[main_page, feedback_page])

pg.run()