from typing import Dict, Any
import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Classifier:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info(f'Initializing {self.__class__.__name__}')
        self.embedder = self._load_embedder()
        self.model = self._load_model()
        self.label_encoder = pickle.load(open('resource/label_encoder.pkl', 'rb'))

    @st.cache(allow_output_mutation=True)
    def _load_model(self):
        return torch.load('resource/classifier.pt', map_location=torch.device('cpu'))

    @st.cache(allow_output_mutation=True)
    def _load_embedder(self):
        return SentenceTransformer('keepitreal/vietnamese-sbert', device='cpu')

    def test_prediction(self):
        """
        Test prediction
        :return: Prediction result
        """
        batch = ['áo choàng đông', ' iphone 13 promax']
        self.log.info(f"Predicting batch sample : {batch}")
        return self.predict(batch)

    def predict(self, batch: list = []) -> dict[str, str]:
        assert type(batch) == list, "Batch must be a list"
        batch_embedded = self.embedder.encode(batch, convert_to_tensor=True, batch_size=min(len(batch), 2048))
        self.model.eval()
        with torch.no_grad():
            out_data = self.model(batch_embedded)
            ps = torch.exp(out_data)
            pred = ps.max(1).indices.cpu().numpy()
            res = [self.label_encoder.inverse_transform([i])[0] for i in pred]

        return dict(zip(batch, res))


if __name__ == "__main__":
    classifier = Classifier()
    print(classifier.test_prediction())
