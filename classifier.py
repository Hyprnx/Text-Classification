from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer
import logging
import warnings
from time import time
import onnx
import onnxruntime as ort

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class BaseClassifier:
    def __init__(self):
        begin = time()
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info(f'Initializing {self.__class__.__name__} >>>')
        self.embedder = self._load_embedder()
        self.label_encoder = self._load_label_encoder()
        self.model = self._load_model()
        self.log.info(f'Initialized {self.__class__.__name__} accomplished in {time() - begin} seconds')

    def _load_embedder(self):
        raise NotImplementedError("Implement in child class")

    def _load_model(self):
        raise NotImplementedError("Implement in child class")

    def _load_label_encoder(self):
        raise NotImplementedError("Implement in child class")


class Classifier(BaseClassifier):
    def __init__(self):
        super(Classifier, self).__init__()

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

        total_pred_time = time() - begin

        return dict(zip(batch, res)), total_pred_time


class ONNXClassifier(BaseClassifier):
    def __init__(self):
        self.path = 'resource/classifier.onnx'
        super().__init__()
        onnx.checker.check_model(onnx.load(self.path))
        self.model_label_name=self.model.get_outputs()[0].name
        self.model_input_name=self.model.get_inputs()[0].name

    # @st.cache(allow_output_mutation=True)
    def _load_model(self) -> ort.InferenceSession:
        """
        Load ONNX session with cache to prevent reloading model after predicting
        :return: Onnx session model
        """
        self.log.info("Initializing ONNX session>>>")
        so = ort.SessionOptions()
        so.add_session_config_entry('session.load_model_format', 'ONNX')
        ort_sess = ort.InferenceSession(self.path, providers=['CPUExecutionProvider'], sess_options=so)

        if ort_sess:
            self.log.info(f"Neural Net Initialized in {ort_sess.get_profiling_start_time_ns()} ns")
        else:
            self.log.error("Neural Net Initialization Failed")

        return ort_sess

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

    def test_prediction(self) -> tuple[dict[str, str], float]:
        """
        Test prediction
        :return: Prediction result
        """
        batch = ['áo choàng đông', ' iphone 13 promax']
        self.log.info(f"Predicting batch sample : {batch}")
        return self.predict(batch)

    def predict(self, batch: list) -> tuple[dict[str, str], float]:
        """
        Predict batch of sentences
        :param batch: list of text that need to be predicted
        :return: a dictionary of text and its predicted label
        """
        assert type(batch) == list, "Batch must be a list"
        begin = time()
        batch_embedded = self.embedder.encode(batch, batch_size=min(len(batch), 2048))
        pred = self.model.run([self.model_label_name], {'text_embedding': batch_embedded})
        res = [self.label_encoder.inverse_transform([i])[0] for i in [pred[0][index].argmax(0)for index, _ in enumerate(pred[0])]]
        self.log.info(res)
        total_pred_time = time() - begin
        return dict(zip(batch, res)), total_pred_time


if __name__ == "__main__":
    classifier = ONNXClassifier()
    print(classifier.test_prediction())
    classifier = Classifier()
    print(classifier.test_prediction())



