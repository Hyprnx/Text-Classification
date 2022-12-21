import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("The \" Production Ready\" process")


st.markdown("""
    Ever wondered how to tell if an application is ready for production? Well, let’s see what these principles are 
    and how they can make your life easy.

    Production readiness refers to when a certain application or a program will be ready to operate. Once the 
    application is made, we call it a production-ready application. At this point, the application should be capable 
    of handling production-level traffic, data and security.

    Once an application is marked production-ready, it can be trusted completely to handle its real work. Following 
    the principles of availability, the application should be available for all the intended users.
""")

st.markdown("""
    ## The Five Production Readiness Principles
    - Stability and Reliability
    - Scalability and Performance
    - Fault Tolerance and Disaster Recovery
    - Monitoring
    - Documentation
""")



next_page = st.button("Demo")
if next_page:
    switch_page("Demo")
