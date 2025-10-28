import os
import unicodedata

if not os.getenv("FTLANG_CACHE"):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(current_dir)
    ftlang_cache_dir = os.path.join(root_dir, 'resources', 'fasttext-langdetect')
    os.environ["FTLANG_CACHE"] = str(ftlang_cache_dir)
    # print(os.getenv("FTLANG_CACHE"))

from fast_langdetect import detect_language


def remove_invalid_surrogates(text):

    return ''.join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def detect_lang(text: str) -> str:

    if len(text) == 0:
        return ""

    text = text.replace("\n", "")
    text = remove_invalid_surrogates(text)

    # print(text)
    try:
        lang_upper = detect_language(text)
    except:
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        lang_upper = detect_language(html_no_ctrl_chars)

    try:
        lang = lang_upper.lower()
    except:
        lang = ""
    return lang