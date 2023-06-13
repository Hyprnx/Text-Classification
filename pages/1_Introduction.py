import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("Greeting, this is Product name classification for To Duc Anh Graduation Thesis introduction page.")

st.markdown(
    """
        Welcome to my classification module, which was created to correctly categorize products on e-commerce websites. 
        This module provides efficient and effective product categorization to improve the data management and analytics 
        process by utilizing the latest techniques in machine learning and deep learning. I'm thrilled to be able to share 
        my work and hope that it will help to streamline and improve e-commerce operations.
    """)

st.subheader(
    """
        About the author:
    """)

st.markdown(
    """
    I am proud to be a Management Associate at Techcombank Vietnam and a research assistant at the National Economics University 
    Data Science Laboratory as a driven and multi-talented individual. My experience in the financial industry and academia has 
    given me a diverse skill set as well as a passion for using data and technology to drive growth and improve processes. 
    I am committed to constantly learning and expanding my knowledge of data science and management.
    """)


st.markdown("""
    ##### In this sample website, I will introduce how we clean the data, build the model, show how the model is
     production ready, future plan and finally, we will show a demo of our model.
""")

next_page = st.button("How I did it")
if next_page:
    switch_page("How I did it")
    
st.markdown("""[Source code](https://github.com/Hyprnx/Text-Classification)""")
