import gc
import pandas as pd
import streamlit as st
from classifier import ONNXClassifier, TorchClassifier
from normalizer import dataframe_normalize

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )


@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


st.header("Demo with a sample batch")
st.markdown("""
    #### Interactive Prediction
    You can try with one, or many samples enter your sample like below, each sample are separated by a comma.
    The result will be shown after finished running.
    That's it, let's try!
""")

# Model Selection
st.radio(
    "Choose your classifier",
    key="model",
    options=["PyTorch model with pretrained Vietnamese sBERT", "Converted ONNX model with pretrained Vietnamese sBERT"],
)

model_dict = {
    "PyTorch model with pretrained Vietnamese sBERT": TorchClassifier(),
    "Converted ONNX model with pretrained Vietnamese sBERT": ONNXClassifier(),
}

# Data Inference
# Sentence wise inference
sentence = st.text_input('Input your sentence here:', placeholder="áo choàng đông, iphone 13 promax",
                         autocomplete='on')
sentence = sentence.split(',')
if sentence == ['']:
    st.json({"": ""})
    st.write("Prediction accomplished in 0 seconds. Please enter your sentence on text box above")
else:
    classifier = None
    try:
        del classifier
        gc.collect()
    except BaseException:
        pass
    finally:
        classifier = model_dict[st.session_state.model]
        res, time = classifier.predict(sentence)
        st.json(res)
        st.write("Prediction accomplished in {:.4} seconds".format(time))

# Batch inference with csv file
st.markdown("""### Try predict a file:""")
csv_file = st.file_uploader("Upload your csv file here", type=['csv'], accept_multiple_files=False,
                            help="""Only one file can be inference at one, and only csv file is accepted
                            The file must have a column named 'sample' that contains the text to be predicted""")
if not csv_file:
    st.write("Upload file to predict")
else:
    classifier = None
    try:
        del classifier
        gc.collect()
    except BaseException:
        pass
    finally:
        df_ori = pd.read_csv(csv_file, encoding='utf-8')
        df = dataframe_normalize(df_ori)
        classifier = model_dict[st.session_state.model]
        res, time = classifier.predict(df['sample'].tolist(), batch_inference=True)
        del df
        df_ori['predicted_label'] = res
        st.write(f"Model Prediction on {csv_file.name}")
        st.write(df_ori)
        st.write("Prediction accomplished in {:.4} seconds".format(time))

        st.download_button(
            "Download Predicted file",
            convert_df(df_ori),
            f"{csv_file.name.split('.')[0]}_predicted.csv",
            "text/csv",
            key='download-csv'
        )
