"""
util for HW3
"""

import os
import io
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
import unicodedata
import string
import pandas as pd
import numpy as np
import random
import enchant
from geotext import GeoText


def get_documents(path=pre_dir):
    for name in os.listdir(path):
        if name.endswith('.txt'):
            yield os.path.join(path, name)
            
            
printable = set(string.printable)

def clean_corpus(corpus=corpus_dir):
    for path in get_documents():
        with io.open(path, 'r',errors='ignore') as f:
            text = f.read()
            text = filter(lambda x: x in printable, text)
        fname = os.path.splitext(os.path.basename(path))[0] + ".txt"
        outpath = os.path.join(corpus, fname)
        with io.open(outpath, 'w',encoding='utf8',errors='ignore') as f:
            f.write("".join(text))


def is_ordered_sub(sub, lst):
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if all(sub[j] == lst[i+j] for j in range(ln)):
            return True
    return False


def first_occur(sub,lst):
    if len(sub) == 1:
        return lst.index(sub[0])
    return lst.index(sub[1]) - 1            

