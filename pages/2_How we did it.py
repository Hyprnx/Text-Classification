import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("This is how we did it")
st.markdown("# Data Normalization!")
WIDTH = 400

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

with st.expander("ℹ️ - phoBERT", expanded=False):
    st.markdown(
        """
        Language model BERT- Bidirectional Encoder Representations from Transformers - is a recent breakthrough in NLP. 
        However, most pre-trained BERT-based models which were learnt using English corpus only, or data combined from 
        different languages, are not aware of the difference between Vietnamese syllables and word tokens. As a result, 
        the pre-trained PhoBERT model was introduced as a state-of-the-art language model for Vietnamese. 
        - **PhoBERT** has two versions, **PhoBERTbase** and **PhoBERTlarge**, using the same architectures of **BERTbase** and **BERTlarge**, respectively. 
        - **PhoBERT** pre-training approach is based on **RoBERTa** which optimizes the BERT pre-training procedure for more robust performance.
        - PhoBERT *outperforms* previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of *Part-of-speech tagging*, *Dependency parsing*, *Named-Entity Recognition(NER)* and *Natural language* inference.
        - PhoBERT can be used with popular open source libraries: *transformers* and *fairseq*.
        """
    )
    st.markdown(r"""
        ### Why phoBERT?
        #### PhoBERT have became a SOTA model for the Vietnamese language model.
        
        Training on a large-scale Vietnamese corpus and employing Vietnamese word segmentation, There are two main 
        concerns in terms of Vietnamese language modeling:
        - The Vietnamese Wikipedia corpus which is relatively small (1GB in size uncompressed) is the only data used to train monolingual language models.
        - All publicly released monolingual and multilingual BERT-based language models are not aware of the difference between Vietnamese syllables and word tokens. Without doing a pre-process step of Vietnamese word segmentation, those models directly apply Byte-Pair encoding (BPE) methods to the syllable-level Vietnamese pre-training data. Intuitively, for word-level Vietnamese NLP tasks, those models might not perform well
        
        => **PhoBERT to handle previous concerns by**:
        
        - To handle the first problem, the research team trained the first large-scale monolingual BERT-based using a 20GB word-level Vietnamese corpus from Vietnamese Wikipedia corpus (∼1GB), and the second corpus (∼19GB) generated by removing similar articles and duplication from a 50GB Vietnamese news corpus. 
        - To solve the second concern, employ RDRSegmenter (Vietnamese word segmentation) from VnCoreNLP to perform word and sentence segmentation on the pre-training dataset, resulting in ∼145M word-segmented sentences, before going through the BPE encoder.
        
        #### Following RoBERTa's ideological approach, PhoBERT only uses the Masked Language Model task to train, omitting the Next Sentence Prediction task
    """)
    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://phamdinhkhanh.github.io/assets/images/20200523_BERTModel/pic5.png", width=WIDTH)
    with col3:
        st.write()

    st.markdown("""
        PhoBERT architecture (example on essay category)
        Architecture: Pretrained PhoBERT-base and a Softmax regression
    """)
    col1, col2 = st.columns([5, 5])
    with col1:
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4Wg2sz967y4uGnkzX6DeHSZc_zUUBXMCTsg&usqp=CAU',)
    with col2:
        st.image(r"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsbsvKVpSHxFBduh7s9tQu8rMSnORXxq07sA&usqp=CAU",)


with st.expander("ℹ️ - sBERT", expanded=False):
    st.markdown(
        """
        ### sBERT simple introduction
        BERT solves semantic search in a pairwise fashion. It uses a cross-encoder: 2 sentences are passed to BERT 
        and a similarity score is computed. However, when the number of sentences being compared exceeds 
        hundreds/thousands of sentences, this would result in a total of $(n)(n-1)/2$ computations being done.
        """
    )
    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://miro.medium.com/max/786/1*WhtDBvnmtYaujDRCZOIx4w.webp", width=WIDTH)
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
    If you look at the original cross-encoder architecture of **BERT**, **SBERT** is similar to this but removes the 
    final classification head. Unlike BERT, SBERT uses a siamese architecture (as I explained above), where it contains 
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
        st.image(r"https://miro.medium.com/max/720/1*6gjaA_TqojVTABHJPNRMng.webp", width=WIDTH)
    with col3:
        st.write()

    st.markdown("*Source: [An Intuitive Explanation of Sentence-BERT]"
                "(https://towardsdatascience.com/an-intuitive-explanation-of-sentence-bert-1984d144a868)*")

    st.markdown("")

with st.expander("ℹ️ - TFIDF", expanded=False):
    st.markdown("""
        TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of 
        [information retrieval (IR)](https://en.wikipedia.org/wiki/Information_retrieval) and machine learning, that 
        can quantify the importance or relevance of string representations (words, phrases, lemmas, etc)  in a document 
        amongst a collection of documents (also known as a corpus). TF-IDF can be broken down into two parts TF 
        (term frequency) and IDF (inverse document frequency).""")
    st.markdown(r"""
        ### What is TF (term frequency)?
        Term frequency works by looking at the frequency of a particular term you are concerned with relative to the 
        document. There are multiple measures, or ways, of defining frequency:
        - Number of times the word appears in a document (raw count).
        - Term frequency adjusted for the length of the document (raw count of occurences divided by number of words 
        in the document).
        - [Logarithmically scaled](https://en.wikipedia.org/wiki/Logarithmic_scale) frequency (e.g. log(1 + raw count)).
        - [Boolean frequency](https://en.wikipedia.org/wiki/Boolean) (e.g. 1 if the term occurs, or 0 if the term 
        does not occur, in the document).
        
        $$
            TF(t,d) = \frac{\text{Number of times term t appears in a document}}
            {\text{Total number of terms in the document}}
        $$
    """)

    st.markdown(r"""
        ### What is IDF (inverse document frequency)?
        Inverse document frequency looks at how common (or uncommon) a word is amongst the corpus. IDF is calculated 
        as follows where $t$ is the term (word) we are looking to measure the commonness of and $N$ is the number of 
        documents $(d)$ in the corpus $(D)$.. The denominator is simply the number of documents in which the term, $t$
        , appears in.
        $$
            IDF(t,D) = \log \frac{\text{N}}{count(d \in D: t \in d)}
        $$
        The reason we need IDF is to help correct for words like “of”, “as”, “the”, etc.. since they appear frequently 
        in an English corpus, similar to Vietnamese, there are words such as “là”, “của”, “cứ”…. Thus by taking 
        inverse document frequency, we can minimize the weighting of frequent terms while making infrequent terms have 
        a higher impact. Finally IDFs can also be pulled from either a background corpus, which corrects for sampling 
        bias, or the dataset being used in the experiment at hand.
    """)

    st.markdown(r"""
        ### Putting it together: TF-IDF
        To summarize the key intuition motivating TF-IDF is the importance of a term is inversely related to its 
        frequency across documents.TF gives us information on how often a term appears in a document and IDF gives us 
        information about the relative rarity of a term in the collection of documents. By multiplying these values 
        together we can get our final TF-IDF value.
        $$
            TF-IDF(t,d,D) = TF(t,d) \times IDF(t,D)
        $$
        The higher the TF-IDF score the more important or relevant the term is; as a term gets less relevant, 
        its TF-IDF score will approach 0.
    """)

    st.markdown("""
        ### Using TF-IDF in machine learning & natural language processing
        Machine learning algorithms often use numerical data, so when dealing with textual data or any 
        [natural language processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) task, a 
        sub-field of ML/AI dealing with text, that data first needs to be converted to a vector of numerical data by 
        a process known as [vectorization]
        (https://towardsdatascience.com/understanding-nlp-word-embeddings-text-vectorization-1a23744f7223). TF-IDF 
        vectorization involves calculating the TF-IDF score for every word in your corpus relative to that document and 
        then putting that information into a vector (see image below using example documents “A” and “B”). Thus each 
        document in your corpus would have its own vector, and the vector would have a TF-IDF score for every single 
        word in the entire collection of documents. Once you have these vectors you can apply them to various use 
        cases such as seeing if two documents are similar by comparing their TF-IDF vector using 
        [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
    """)

    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://ecm.capitalone.com/WCM/tech/tf-idf-4.png", width=WIDTH)
    with col3:
        st.write()

    st.markdown("""
        ### Pros and cons of using TF-IDF
        #### Pros of using TF-IDF
        The biggest advantages of TF-IDF come from how simple and easy to use it is. It is simple to calculate, it is 
        computationally cheap, and it is a simple starting point for similarity calculations (via TF-IDF vectorization 
        and cosine similarity).
        #### Cons of using TF-IDF
        Something to be aware of is that TF-IDF cannot help carry semantic meaning. It considers the importance of the 
        words due to how it weighs them, but it cannot necessarily derive the contexts of the words and understand 
        importance that way.
        
        TF-IDF ignores word order and thus compound nouns like “Queen of England” will not be considered as a 
        “single unit”. This also extends to situations like negation with “not pay the bill” vs “pay the bill”, 
        where the order makes a big difference. 
        
        Another disadvantage is that it can suffer from memory-inefficiency since TF-IDF can suffer from the curse of 
        dimensionality. Recall that the length of TF-IDF vectors is equal to the size of the vocabulary. In some 
        classification contexts this may not be an issue but in other contexts like clustering this can be unwieldy 
        as the number of documents increases.

    """)

with st.expander("ℹ️ - Count Vectorizer", expanded=False):
    st.markdown(
        """
        ### Count Vectorizer
        To use textual data for predictive modeling, the text must be parsed to remove certain words – this process is 
        called tokenization. These words need to then be encoded as integers, or floating-point values, for use as 
        inputs in machine learning algorithms. This process is called feature extraction (or vectorization).
        
        Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token 
        counts. It also enables the pre-processing of text data prior to generating the vector representation. This 
        functionality makes it a highly flexible feature representation module for text.
        """
    )

    col1, col2, col3 = st.columns([1, 5, 0.2])
    with col1:
        st.write('')
    with col2:
        st.image(r"https://i.imgur.com/7wOovio.png", width=WIDTH)
    with col3:
        st.write()

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
