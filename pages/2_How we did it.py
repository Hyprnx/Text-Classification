import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("This is how we did it")
st.markdown("# Data Normalization!")

st.markdown("""
    ## About the dataset
    The data we used were crawled from the ecomerce site [Shopee](https://shopee.vn/). The data were in the form of a
    csv file with 2 columns: product name and category. The dataset contains 800 thousand products with 4 categories,
    which are:
    - Electronics related (Điện Tử - Điện Máy)
    - Fashion (Thời Trang)
    - Mom and Baby (Mẹ và Bé)
    - Cosmetics (Mỹ Phẩm)
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
        We did try a lot of text embedding methods, since calculating text embedding can be done in various methods. 
        But the methods should be fast, reliable, scalable and perhap should be paralellizable. We did try the 
        following methods: 
        - [Vietnamese Sentence Bi-directional Encoder Representations from Transformers ( Vietnamese sBERT)](https://huggingface.co/keepitreal/vietnamese-sbert) from HuggingFace.
        - [phoBERT](https://huggingface.co/vinai/phobert-base) from HuggingFace.
        - [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from Scikit-learn.
        - [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from Scikit-learn.
""")
with st.expander("ℹ️ - sBERT", expanded=False):
    st.markdown(
        """
        ### sBERT simple introduction?
        BERT solves semantic search in a pairwise fashion. It uses a cross-encoder: 2 sentences are passed to BERT 
        and a similarity score is computed. However, when the number of sentences being compared exceeds 
        hundreds/thousands of sentences, this would result in a total of $(n)(n-1)/2$ computations being done.
        """
    )
    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://miro.medium.com/max/786/1*WhtDBvnmtYaujDRCZOIx4w.webp", width=400)
    with col3:
        st.write()

    st.markdown(
        """
        The BERT cross-encoder consists of a standard BERT model that takes in as input the two sentences, 
        A and B, separated by a [SEP] token. On top of the BERT is a feedforward layer that outputs a 
        similarity score.
        """)

    st.markdown(
        """ To overcome this problem, researchers had tried to use BERT to create sentence embeddings. 
            The most common way was to input individual sentences to BERT — and remember that BERT computes 
            word-level embeddings, so each word in the sentence would have its own embedding. After the 
            sentences were inputted to BERT, because of BERT’s word-level embeddings, the most common way to 
            generate a sentence embedding was by averaging all the word-level embeddings or by using the 
            output of the first token (i.e. the [CLS] token). However, this method often resulted in bad 
            sentence embeddings, often averaging worse than 
            averaged *[GLoVE](https://nlp.stanford.edu/projects/glove/)* embeddings.
            """)
    st.markdown(
        """
            Essentially, the SBERT network uses a concept called 
            [Triplet Loss](https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)
            to train its siamese architecture.
        """)

    st.markdown("""
    ### What does SBERT do and how does it work?
    If you look at the original cross-encoder architecture of BERT, SBERT is similar to this but removes the final 
    classification head. Unlike BERT, SBERT uses a siamese architecture (as I explained above), where it contains 
    2 BERT architectures that are essentially identical and share the same weights, and SBERT processes 2 sentences 
    as pairs during training.
    
    Let’s say that we feed sentence A to BERT A and sentence B to BERT B in SBERT. Each BERT outputs pooled sentence 
    embeddings. While the original research paper tried several pooling methods, they found mean-pooling was the best 
    approach. Pooling is a technique for generalizing features in a network, and in this case, mean pooling works by 
    averaging groups of features in the BERT.
    
    After the pooling is done, we now have 2 embeddings: 1 for sentence A and 1 for sentence B. When the model is 
    training, SBERT concatenates the 2 embeddings which will then run through a softmax classifier and be trained 
    using a softmax-loss function. At inference — or when the model actually begins predicting — the two embeddings 
    are then compared using a cosine similarity function, which will output a similarity score for the two sentences. 
    Here is a diagram for SBERT when it is fine-tuned and at inference.""")

    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://miro.medium.com/max/720/1*6gjaA_TqojVTABHJPNRMng.webp", width=400)
    with col3:
        st.write()

    st.markdown("*Source: [An Intuitive Explanation of Sentence-BERT]"
                "(https://towardsdatascience.com/an-intuitive-explanation-of-sentence-bert-1984d144a868)*")

    st.markdown("")

with st.expander("ℹ️ - phoBERT", expanded=False):
    st.write(
        """
        Place-holder for phoBERT
        """
    )
    st.markdown("")

with st.expander("ℹ️ - TFIDF", expanded=False):
    st.write(
        """
        Place-holder for TFIDF
        """
    )
    st.markdown("")

with st.expander("ℹ️ - Count Vectorizer", expanded=False):
    st.write(
        """
        Place-holder for Count Vectorizer
        """
    )
    st.markdown("")

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


next_page = st.button("The Production-Ready part")
if next_page:
    switch_page("The Production-Ready part")
