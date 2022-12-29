from common.check_file_exist import is_file_empty
import pickle
import gdown

from normalizer import norm

OUTPUT = 'resource/Vectorizer.pkl'
DRIVE_PATH = 'https://drive.google.com/u/0/uc?id=1y0-EEz2lifgRXgzYsgwrK6CLE7ikrLPE&export=download&confirm=t&uuid=0ffc1320-c579-402f-af77-381fe06bf76d&at=ACjLJWnMw-2_iWKAAfLdJ8s5Tfbf:1672304225283'


def load_vectorizer():
    if not is_file_empty(OUTPUT):
        return pickle.load(open(OUTPUT, 'rb'))
    else:
        gdown.download(DRIVE_PATH, OUTPUT, quiet=False, use_cookies=False)
        return pickle.load(open(OUTPUT, 'rb'))


if __name__ == '__main__':
    vectorizer = load_vectorizer()
    print(dir(vectorizer))
    print(vectorizer.get_params())
    vectorized_txt = vectorizer.transform([norm('Hello World')])
    print(vectorized_txt.shape)

