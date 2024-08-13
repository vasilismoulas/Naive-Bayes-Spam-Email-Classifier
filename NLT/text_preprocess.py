import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

from enum import IntEnum


class CASE(IntEnum):
    LOWER = 0
    UPPER = 1

def tokenizer(text: str) -> str:
    '''
    Tokenize string.
    '''
    text = word_tokenize(text)

    return text

def case_tranformation(tokens: list, case: CASE) -> list:
    '''
    Transform case(e.g.: LOWER = 0 or UPPER = 1).
    '''
    if case == CASE.LOWER:
       lower_tokens = [token.lower() for token in tokens]
    else:
       lower_tokens = [token.upper() for token in tokens] 

    return lower_tokens

def remove_punctuation(tokens: list) -> list:
    '''
    Remove punctuation.
    '''
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]

    return cleaned_tokens
