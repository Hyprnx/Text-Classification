import streamlit as st

st.header("This is how we did it")
st.markdown("# Data Normalization!")

st.markdown("""
    ## About the dataset
    The data we used were crawled from the ecomerce site [Shopee](https://shopee.vn/). The data were in the form of a
    csv file with 2 columns: product name and category. 
    """)


st.markdown("""
    ## The problem
    ### Different text font
    The online ecommerce platform consists of many seller, they be there, with a ultimate goal, is to sell as much
    product as they possibly can. So that they will try to make their product as much appealling as possible. 
    For example, like this:
    
    - 𝑺ữ𝒂 𝒓ử𝒂 𝒎ặ𝒕 𝑪𝒆𝒓𝒂𝒗𝒆 236𝒎𝒍, 355𝒎𝒍, 473𝒎𝒍 𝒄𝒉𝒐 𝒅𝒂 𝒅ầ𝒖, 𝒅𝒂 𝒌𝒉ô
    - 𝚅ò𝚖 𝙲ả𝚖 Ứ𝚗𝚐, 𝚅ò𝚖 𝙾𝚖𝚎𝚐𝚊 𝙳𝚎𝚟𝚘𝚒, 𝚋ổ 𝚜𝚞𝚗𝚐 á𝚗𝚑 𝚜á𝚗𝚐 𝙲𝚊𝟸+, 𝚌𝚑ố𝚗𝚐 𝚘𝚡𝚢 𝚑ó𝚊 𝙷𝚢𝚍𝚛𝚘𝚐𝚎𝚗
    
    These text are human-readable, we can translate the text to normal text just fine. But machine, they cant, every
    text will be translate to some type of unicode code, and these font above, is completely different from the normal
    text we use.
    
    ### Additional infomation and numbers
    More over, sellers even add additional infomation to the product name, like this:
    - [𝐂𝐇Í𝐍𝐇 𝐇Ã𝐍𝐆 💯%] 𝐁𝐎𝐃𝐘 𝐌Ề𝐌 𝐍ƯỚ𝐂 𝐇𝐎𝐀 𝐀𝐂𝐎𝐒𝐌𝐄𝐓𝐈𝐂 𝟐𝟎𝟐𝟎
    - 【﻿[　０７／２０２３　ＳＡＬＥ　ＳỐＣ　]　Ｓữａ　ｒửａ　ｍặｔ　７　ｖị　ＥＣＯＳＹ　Ｈàｎ　Ｑｕốｃ】
    
    The different type of text, give human more infomation, but that is the infomation about the product, not their 
    categories, since whether the product is authentic or not, or it is on sale or not or how many products are in a 
    pack, does not matter with product category task, with both human and computer.""")


st.markdown("""
    ## Solution
    The solution is to build a preprocessor that can normalize the sample name to the type that we can use to train our
    model. The preprocessor will do the following:
    - Remove the additional infomation in bracket.
    - Normalize the unicode code using the [unicodedata](https://docs.python.org/3/library/unicodedata.html) library.
    
    The result show look like the following:
    
    
    |                |Original text                  |Text after preprocessed      |
    |----------------|-------------------------------|-----------------------------|
    |Product name	 |`𝑺ữ𝒂 𝒓ử𝒂 𝒎ặ𝒕 𝑪𝒆𝒓𝒂𝒗𝒆 236𝒎𝒍, 355𝒎𝒍, 473𝒎𝒍 𝒄𝒉𝒐 𝒅𝒂 𝒅ầ𝒖, 𝒅𝒂 𝒌𝒉ô` |"Sữa rửa mặt Cerave cho da dầu khô"|
    |	             |`𝚅ò𝚖 𝙲ả𝚖 Ứ𝚗𝚐, 𝚅ò𝚖 𝙾𝚖𝚎𝚐𝚊 𝙳𝚎𝚟𝚘𝚒, 𝚋ổ 𝚜𝚞𝚗𝚐 á𝚗𝚑 𝚜á𝚗𝚐 𝙲𝚊𝟸+, 𝚌𝚑ố𝚗𝚐 𝚘𝚡𝚢 𝚑ó𝚊 𝙷𝚢𝚍𝚛𝚘𝚐𝚎𝚗`|"Vòm Cảm Ứng Vòm Omega Devoi bổ sung ánh sáng chống oxy hóa Hydrogen"|
    |				 |`-[𝐂𝐇Í𝐍𝐇 𝐇Ã𝐍𝐆 💯%] 𝐁𝐎𝐃𝐘 𝐌Ề𝐌 𝐍ƯỚ𝐂 𝐇𝐎𝐀 𝐀𝐂𝐎𝐒𝐌𝐄𝐓𝐈𝐂 𝟐𝟎𝟐𝟎- `|"Body mềm nước hoa ACOSMETIC"|
    
    """)

st.markdown("# Calculate Text Embeddings!")
st.markdown("""
    ## What is text embedding?
    Computer can only understand numbers, so we need to convert text to some sort of numbers, and that is what text embedding is for.
    Text embedding is a way to represent text in a vector space, so that we can use the vector to train our model.
""")

st.markdown("""
        ## What text embedding method did we use?
        We did try a lot of text embedding methods, since calculating text embedding can be done in various methods. But the methods should be
        fast, reliable, scalable and perhap should be paralellizable. We did try the following methods: 
        - [Vietnamese Sentence Bi-directional Encoder Representations from Transformers ( Vietnamese sBERT)](https://huggingface.co/keepitreal/vietnamese-sbert) from HuggingFace.
        - [phoBERT](https://huggingface.co/vinai/phobert-base) from HuggingFace.
        - [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from Scikit-learn.
        - [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from Scikit-learn.
""")

st.markdown("# Build, train and validate models!")
st.markdown("""
    ## What model did we use?
    We expermented with a lot of models, but the models should be fast, reliable, and scalable. We did try the following models:
    - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), 
        [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html),
        [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html),
        [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html),
        [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from Scikit-learn.
    - Custom build Neural Network using [Tensorflow Keras](https://keras.io/api/models/sequential/) and [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
""")

st.markdown("""
    ## The result
    All model tested on different type of text embedding method, and the result is not much different, archiving high
     8x% to 9x% accuracy. In our opinion, choosing what model is not really important, since the result is not 
     much different, but choosing the right model that ready for production environment is. 
""")
