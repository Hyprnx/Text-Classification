import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )


st.title("Deep Learning Final Project")

st.image(
    "https://images.unsplash.com/photo-1541854615901-93c354197834?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8f"
    "GVufDB8fHx8&auto=format&fit=crop&w=2073&q=80",
    width=600,
)

st.subheader(
    """
        This is Text Classification project for the Deep Learning course at DSEB61@NEU.
    """
)

with st.expander("ℹ️ - About this page", expanded=True):
    st.write(
        """
        This web page is purely for demonstration purpose. The model is trained on a dataset of 800 thousand products 
        for the Final Project of Deep Learning course. We will not take any responsibility for the result of the model.
        Thank you for your understanding.
        """
    )
    st.markdown("")


st.subheader(
    """
        The content today are:
    """
)

introduction = st.button("Introduction")
if introduction:
    switch_page("Introduction")

process = st.button("How we did it")
if process:
    switch_page("How we did it")

prod_ready = st.button("The Production-Ready part")
if prod_ready:
    switch_page("The Production-Ready part")

demo = st.button("Demo")
if introduction:
    switch_page("Demo")