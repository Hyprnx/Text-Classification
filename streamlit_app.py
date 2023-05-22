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
    width=700,
)

st.subheader(
    """
        Greeting, this is Product name classification for To Duc Anh Graduation Thesis introduction page.
    """
)

with st.expander("ℹ️ - About this page", expanded=True):
    st.write(
        """
        Welcome to the project! This website is the result of my graduation thesis and is dedicated to providing a 
        novel approach to product classification. I intend to accurately classify products based solely on their 
        product names by employing cutting-edge deep learning techniques. My ambition is to make product classification 
        simpler and more efficient for both businesses and individuals. Thank you for visiting, and I hope you find our 
        website informative and useful in better categorizing products on e-commerce websites.
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

process = st.button("How I did it")
if process:
    switch_page("How I did it")

prod_ready = st.button("The Production-Ready part")
if prod_ready:
    switch_page("The Production-Ready part")

demo = st.button("Demo")
if demo:
    switch_page("Demo")