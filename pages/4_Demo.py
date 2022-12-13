import streamlit as st
from classifier import Classifier

classifier = Classifier()

st.header("Demo")

sentence = st.text_input('Input your sentence here:')
sentence = sentence.split(",")
print(sentence)

if sentence:
    st.write(classifier.predict(sentence))