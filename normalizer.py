import re
from unicodedata import normalize
import pandas as pd
import visen
from time import time

opening_ls = ['[', '{', '⁅', '〈', '⎡', '⎢', '⎣', '⎧', '⎨', '⎩', '❬', '❰', '❲', '❴', '⟦', '⟨', '⟪', '⟬', '⦃', '⦇', '⦉',
              '⦋', '⦍', '⦏', '⦑', '⦓', '⦕', '⦗', '⧼', '⸂', '⸄', '⸉', '⸌', '⸜', '⸢', '⸤', '⸦', '〈', '《', '「', '『',
              '【', '〔', '〖', '〘', '〚', '﹛', '﹝', '［', '｛', '｢', '｣']

closing_ls = [']', '}', '⁆', '〉', '⎤', '⎥', '⎦', '⎫', '⎬', '⎭', '❭', '❱', '❳', '❵', '⟧', '⟩', '⟫', '⟭', '⦄', '⦈', '⦊',
              '⦌', '⦎', '⦐', '⦒', '⦔', '⦖', '⦘', '⧽', '⸃', '⸅', '⸊', '⸍', '⸝', '⸣', '⸥', '⸧', '〉', '》', '」', '』',
              '】', '〕', '〗', '〙', '〛', '﹜', '﹞', '］', '｝', '｣']

opening_bracket = {key: '(' for key in opening_ls}
closing_bracket = {key: ')' for key in closing_ls}

opening_bracket_pattern = {f"\\{key}": "(" for key in opening_ls}
closing_bracket_pattern = {f"\\{key}": ")" for key in closing_ls}

PUNC = '!\"#$&()*+,-–−./:;=?@[\]^_`{|}~”“`°²ˈ‐ㄧ‛∼’'  # remove <> for number_sym and unknown_sym
re_num_and_decimal = '[0-9]*[,.\-]*[0-9]*[,.\-]*[0-9]*[.,\-]*[0-9]*[,.\-]*[0-9]+[.,]?'
re_unknown = '[a-z]+[\d]+[\w]*|[\d]+[a-z]+[\w]*'
re_vnese_txt = r'[^a-z0-9A-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉ' \
               r'ẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s|_]'
special_punc = {'”': '"', '': '', "’": "'", "`": "'"}


def replace_all(replacer: dict, txt: str) -> str:
    """
    Replace all the keys in the dictionary with their respective values.
    :param replacer: dictionary of keys are the words to be replaced and values are the words to replace them with
    :param txt: subject string
    :return: string after replacement
    """
    for old, new in replacer.items():
        txt = txt.replace(old, new)
    return txt


def replace_num(txt: str) -> str:
    """
    Replace all the numbers in the text with blank
    :param text: subject string
    :return: string after replacement
    """
    text = re.sub(re_num_and_decimal, '', txt)
    return text


def replace_unknown(text: str) -> str:
    """
    Replace all the predefined unknown symbols in the text with blank
    :param text: subject string
    :return: string after replacement
    """
    text = re.sub(re_unknown, '', text)
    return text


def unicode_normalizer(text, forms: list = ['NFKC', 'NKFD', 'NFC', 'NFD']) -> str:
    """
    Normalize unicode text
    :param text: subject string
    :param forms: unicode normalization forms
    :return: string after normalization
    """
    for form in forms:
        text = normalize(form, text)
    return text


def normalize_bracket(text: str) -> str:
    """
    Normalize brackets in the text with predefined string for later use
    :param text: subject string
    :return: transformed string
    """
    text = replace_all(opening_bracket, text)
    text = replace_all(closing_bracket, text)
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    return text


def remove_punc(text: str) -> str:
    """
    Remove punctuations in the text
    :param text: subject string
    :return: string after removal
    """
    r = re.compile(r'[\s{}]+'.format(re.escape(PUNC)))
    text = r.split(text)
    return ' '.join(i for i in text if i)


def norm(text: str) -> str:
    """
    Normalize text by removing punctuations, numbers, unknown symbols, brackets, and normalize unicode
    :param text: subject string
    :return: normalized string
    """
    text = str(text)
    text = text.lower()
    text = text.split('\n')[0]
    text = unicode_normalizer(text, ["NFKC", "NFKD", "NFD", "NFC"])
    text = replace_all(special_punc, text)
    text = normalize_bracket(text)
    text = replace_unknown(text)
    text = replace_num(text)
    text = remove_punc(text)
    text = re.sub(re_vnese_txt, "", text)
    text = text.strip()
    return visen.clean_tone(text)


def dataframe_normalize(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Normalize dataframe
    :param df: Dataframe to normalize
    :return: Dataframe normalized
    """
    assert 'sample' in df.columns, f"DataFrame with column name 'sample' expected, got {df.columns} "
    begin = time()

    df['sample'] = df['sample'].str.lower()
    df['sample'] = df['sample'].str.replace('[a-z]+[\d]+[\w]*|[\d]+[a-z]+[\w]*', '',
                                            regex=True)  # replace all product code with blank
    df['sample'] = df['sample'].str.replace('[0-9]*[,.\-]*[0-9]*[,.\-]*[0-9]*[.,\-]*[0-9]*[,.\-]*[0-9]+[.,]?', '',
                                            regex=True)  # replace all number with blank
    df['sample'] = df['sample'].replace(opening_bracket_pattern, regex=True)  # normalize opening brackets, listed above
    df['sample'] = df['sample'].replace(closing_bracket_pattern, regex=True)  # normalize closing brackets, listed above
    df['sample'] = df['sample'].str.replace("[\(\[].*?[\)\]]", '',
                                            regex=True)  # remove all content inside brackets, eg: (Siêu sale 12/12)
    df['sample'] = df['sample'].str.normalize('NFKC')  # Normalize unicode with NFKC standard
    df['sample'] = df['sample'].str.normalize('NFKD')
    df['sample'] = df['sample'].str.normalize('NFC')
    df['sample'] = df['sample'].str.normalize('NFD')
    #   # Remove non Vietnamese text, seems like unessesary right now, might consider remove later to improve run time.
    df['sample'] = df['sample'].str.replace(
        r"[^a-z0-9A-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s|_]",
        '', regex=True)
    df['sample'] = df['sample'].str.replace('đ',
                                            'd').str.strip()  # Special case where unicode normalizer does not consider 'đ' a combination of base unicode characters
    print('Total enlapsed time:', time() - begin, 'seconds')
    return df


if __name__ == '__main__':
    sample_txts = [
        '〖𝔰𝔬𝔪𝔢 𝔣𝔞𝔫𝔠𝔶 𝔭𝔯𝔬𝔡𝔲𝔠𝔱 𝔫𝔞𝔪𝔢〗',
        '𝖘𝖔𝖒𝖊 𝖋𝖆𝖓𝖈𝖞 𝖕𝖗𝖔𝖉𝖚𝖈𝖙 𝖓𝖆𝖒𝖊',
        '❰some product code which will overfit this sample❱ 𝓼𝓸𝓶𝓮 𝓯𝓪𝓷𝓬𝔂 𝓹𝓻𝓸𝓭𝓾𝓬𝓽 𝓷𝓪𝓶𝓮',
        '𝓈𝑜𝓂𝑒 𝒻𝒶𝓃𝒸𝓎 𝓅𝓇𝑜𝒹𝓊𝒸𝓉 𝓃𝒶𝓂𝑒',
        '{𝕤𝕠𝕞𝕖 𝕗𝕒𝕟𝕔𝕪 𝕡𝕣𝕠𝕕𝕦𝕔𝕥 𝕟𝕒𝕞𝕖',
        '☯😝ｓｏｍｅ ｆａｎｃｙ ｐｒｏｄｕｃｔ ｎａｍｅ☯😝',
        '𝐬𝐨𝐦𝐞 𝐟𝐚𝐧𝐜𝐲 𝐩𝐫𝐨𝐝𝐮𝐜𝐭 𝐧𝐚𝐦𝐞',
        '𝘀𝗼𝗺𝗲 𝗳𝗮𝗻𝗰𝘆 𝗽𝗿𝗼𝗱𝘂𝗰𝘁 𝗻𝗮𝗺𝗲',
        '𝘴𝘰𝘮𝘦 𝘧𝘢𝘯𝘤𝘺 𝘱𝘳𝘰𝘥𝘶𝘤𝘵 𝘯𝘢𝘮𝘦',
        '▄▀▄▀▄▀ 𝙨𝙤𝙢𝙚 𝙛𝙖𝙣𝙘𝙮 𝙥𝙧𝙤𝙙𝙪𝙘𝙩 𝙣𝙖𝙢𝙚 ▄▀▄▀▄▀',
        '🌌  🎀 𝚜𝚘𝚖𝚎 𝚏𝚊𝚗𝚌𝚢 𝚙𝚛𝚘𝚍𝚞𝚌𝚝 𝚗𝚊𝚖𝚎🌌  🎀 ',
        '(っ◔◡◔)っ ♥ some fancy product name ♥',
    ]

    for sample in sample_txts:
        print(f'sample \"{sample}\", after normalize: \"{norm(sample)}\"')
