import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("Hello, this is our final project for Deep Learning course")

st.subheader("We are going to build a model to predict the category of a product based on its name")

st.header("Our team")
st.subheader("We are a team of 4 students from the National Economics University, Hanoi, with the major of "
             "Data Science in Economics and Business, class of 2023.")

st.subheader("Our team members are:")
st.write("1. To Duc Anh - Lead")
st.write("2. Nguyen Viet Duong")
st.write("3. Dinh Thi Ha Phuong")
st.write("4. Nguyen Hoai Linh")

st.markdown("""
    ##### In this sample website, we will introduce how we clean the data, build the model, show how the model is
     production ready, future plan and finally, we will show a demo of our model.
""")

next_page = st.button("How we did it")
if next_page:
    switch_page("How we did it")
