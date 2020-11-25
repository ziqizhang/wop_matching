import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os,csv
from copy import deepcopy
import string

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def tokenize(words):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return []
    words = str(words)
    words = words.replace('&amp;', '')
    words = words.replace('&reg;', '')
    words = words.replace('&quot;', '')
    words = words.replace('\t;', ' ')
    words = words.replace('\n;', ' ')
    return words.lower().translate(str.maketrans('', '', string.punctuation)).split()

def preprocess_string(words, stop_words):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return words
    
    word_list = tokenize(words)
    word_list_stopwords_removed = [x for x in word_list if x not in stop_words]
    words_processed = ' '.join(word_list_stopwords_removed)
    return words_processed

def preprocess_string_column(column):
    stop_words_with_punct = deepcopy(stopwords.words('english'))
    stop_words = list(map(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)), stop_words_with_punct))
    
    column = column.apply(preprocess_string, args=(stop_words,))
        
    return column

#this method processes the orignally downloaded DM EL datasets from: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
# and save them into the right format that DM can use
# WARNING: data in the input folder will be overwritten!!!
def parsse_dm_datasets(inFolder, delimiter=','):
    tableA = pd.read_csv(inFolder+"/tableA.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('')

    header= list(tableA.columns.values)

    tableA=tableA.to_numpy()
    d_tableA={}
    for row in tableA:
        d_tableA[row[0]]=row

    tableB = pd.read_csv(inFolder + "/tableB.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    d_tableB = {}
    for row in tableB:
        d_tableB[row[0]] = row

    train = pd.read_csv(inFolder + "/train.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    with open(inFolder + "/train.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        new_header=["id","label"]
        for h in header:
            new_header.append("Left_"+h)
        for h in header:
            new_header.append("Right_" + h)
        writer.writerow(list(new_header))

        id=0
        for row in train:
            id+=1
            l=row[0]
            r=row[1]
            label=row[2]

            ltable=d_tableA[l]
            rtable=d_tableB[r]
            new_row=[id,label]

            for i in range(1, len(ltable)):
                new_row.append(ltable[i])
            for i in range(1, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)


    test = pd.read_csv(inFolder + "/test.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('').to_numpy()
    with open(inFolder + "/test.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        new_header = ["id", "label"]
        for h in header:
            new_header.append("Left_" + h)
        for h in header:
            new_header.append("Right_" + h)
        writer.writerow(list(new_header))

        id = 0
        for row in test:
            id += 1
            l = row[0]
            r = row[1]
            label = row[2]

            ltable = d_tableA[l]
            rtable = d_tableB[r]
            new_row = [id, label]

            for i in range(1, len(ltable)):
                new_row.append(ltable[i])
            for i in range(1, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)

    val = pd.read_csv(inFolder + "/valid.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('').to_numpy()
    with open(inFolder + "/valid.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        new_header = ["id", "label"]
        for h in header:
            new_header.append("Left_" + h)
        for h in header:
            new_header.append("Right_" + h)
        writer.writerow(list(new_header))

        id = 0
        for row in val:
            id += 1
            l = row[0]
            r = row[1]
            label = row[2]

            ltable = d_tableA[l]
            rtable = d_tableB[r]
            new_row = [id, label]

            for i in range(1, len(ltable)):
                new_row.append(ltable[i])
            for i in range(1, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)


if __name__ == "__main__":
    #itunes
    parsse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/tmp/Structured/iTunes-Amazon")