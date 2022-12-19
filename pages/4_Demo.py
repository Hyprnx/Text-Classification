import streamlit as st
from classifier import Classifier

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

classifier = Classifier()

st.header("Demo with a sample batch")
st.markdown("""
    #### Interactive Prediction
    You can try with one, or many samples enter your sample like below, each sample are separated by a comma.
    The result will be shown after finished running.
    That's it, let's try!
""")
sentence = st.text_input('Input your sentence here:', placeholder="áo choàng đông, iphone 13 promax",
                         autocomplete='product')


sentence = sentence.split(",")
if sentence == ['']:
    st.json({"": ""})
    st.write("Prediction accomplished in 0 seconds. Please enter your sentence on text box above")

else:
    res, time = classifier.predict(sentence)
    st.json(res)
    st.write("Prediction accomplished in {:.4} seconds".format(time))
