import os


def is_file_empty(file_path):
    """
    :param file_path: path to file
    Check if file is empty by confirming if its size is 0 bytes
    """
    return os.path.isfile(file_path) and os.path.getsize(file_path) == 0


if __name__ == '__main__':
    path = 'resource/Vectorizer.pkl'
    print(os.listdir('resource'))
    print(is_file_empty('resource/Vectorizer.pkl'))
