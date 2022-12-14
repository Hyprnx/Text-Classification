import streamlit as st
from classifier import Classifier

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

classifier = Classifier()

st.header("Demo with a sample batch")
st.subheader("""
    # Interactive Prediction
    You can try with one, or many samples enter your sample like below, each sample are separated by a comma.
    The result will be shown after finished running.
    That's it, let's try!
""")
sentence = st.text_input('Input your sentence here:', placeholder="áo choàng đông, iphone 13 promax")


sentence = sentence.split(",")
print(sentence)

if sentence:
    st.write(classifier.predict(sentence))
