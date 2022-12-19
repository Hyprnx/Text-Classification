from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer
import logging
import warnings
from time import time

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Classifier:
    def __init__(self):
        begin = time()
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info(f'Initializing {self.__class__.__name__}>>>')
        self.embedder = self._load_embedder()
        self.model = self._load_model()
        self.label_encoder = self._load_label_encoder()
        self.log.info(f'Initialized {self.__class__.__name__} accomplished in {time() - begin} seconds')


    @st.cache(allow_output_mutation=True)
    def _load_model(self) -> torch.nn.modules.container.Sequential:
        """
        Load model with cache to prevent reloading model after predicting
        :return: torch model
        """
        self.log.info("Initializing Neural Net>>>")
        return torch.load('resource/classifier.pt', map_location=torch.device('cpu'))

    @st.cache(allow_output_mutation=True)
    def _load_embedder(self) -> SentenceTransformer:
        """
        Load embedder with cache to prevent reloading embedder after predicting
        :return: cached vietnamese-sbert SentenceTransformer
        """
        self.log.info("Initializing embedder>>>")
        return SentenceTransformer('keepitreal/vietnamese-sbert', device='cpu')

    @st.cache(allow_output_mutation=True)
    def _load_label_encoder(self) -> LabelEncoder:
        """
        Load label encoder with cache to prevent reloading label encoder after predicting
        :return: LabelEncoder that have been encoded with class label
        """
        return pickle.load(open('resource/label_encoder.pkl', 'rb'))

    def test_prediction(self):
        """
        Test prediction
        :return: Prediction result
        """
        batch = ['áo choàng đông', ' iphone 13 promax']
        self.log.info(f"Predicting batch sample : {batch}")
        return self.predict(batch)

    def predict(self, batch: list) -> dict[str, str]:
        """
        Predict batch of sentences
        :param batch: list of text that need to be predicted
        :return: a dictionary of text and its predicted label
        """
        assert type(batch) == list, "Batch must be a list"
        begin = time()
        batch_embedded = self.embedder.encode(batch, convert_to_tensor=True, batch_size=min(len(batch), 2048))
        self.model.eval()
        with torch.no_grad():
            out_data = self.model(batch_embedded)
            ps = torch.exp(out_data)
            pred = ps.max(1).indices.cpu().numpy()
            res = [self.label_encoder.inverse_transform([i])[0] for i in pred]

        res['time_taken'] = time() - begin

        return dict(zip(batch, res))


if __name__ == "__main__":
    classifier = Classifier()
    print(classifier.test_prediction())
