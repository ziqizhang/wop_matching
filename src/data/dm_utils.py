'''
code for processing deepmatcher datasets
'''

import pandas as pd
import numpy as np
np.random.seed(42)
import random, gzip, json
random.seed(42)
from pathlib import Path

import os,csv, re
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
def parse_dm_datasets(inFolder, delimiter=','):
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
            if h=="type":
                h="atype"
            new_header.append("left_"+h)
        for h in header:
            if h=="type":
                h="atype"
            new_header.append("right_" + h)
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

            for i in range(0, len(ltable)):
                new_row.append(ltable[i])
            for i in range(0, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)


    test = pd.read_csv(inFolder + "/test.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('').to_numpy()
    with open(inFolder + "/test.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        new_header = ["id", "label"]
        for h in header:
            if h=="type":
                h="atype"
            new_header.append("left_" + h)
        for h in header:
            if h=="type":
                h="atype"
            new_header.append("right_" + h)
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

            for i in range(0, len(ltable)):
                new_row.append(ltable[i])
            for i in range(0, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)

    val = pd.read_csv(inFolder + "/valid.csv", header=0, delimiter=delimiter, quoting=0, encoding="utf-8",
                        ).fillna('').to_numpy()
    with open(inFolder + "/validation.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        new_header = ["id", "label"]
        for h in header:
            if h=="type":
                h="atype"
            new_header.append("left_" + h)
        for h in header:
            if h=="type":
                h="atype"
            new_header.append("right_" + h)
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

            for i in range(0, len(ltable)):
                new_row.append(ltable[i])
            for i in range(0, len(rtable)):
                new_row.append(rtable[i])

            writer.writerow(new_row)

def normalize_name(name):
    n = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', name)
    n = re.sub(r'\W+', ' ', n).strip().lower()
    n=re.sub("([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*)","LETTERNUMBER",n)
    n=re.sub("(?<!\S)\d+(?!\S)","NUMBER",n)
    return n

#method for extracting just the name column from the standard DM datasets = must be the tableA, tableB files
def extract_names(inFolder, name_col):
    print("loading dataset...")
    tableA = pd.read_csv(inFolder + "/tableA.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    tableB = pd.read_csv(inFolder + "/tableB.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    with open(inFolder+"/nameA.txt",'w') as f:
        for row in tableA:
            name=row[name_col]
            name=normalize_name(name).strip()
            if len(name)==0:
                name="NONE"
            f.write(name+"\n")

    with open(inFolder+"/nameB.txt",'w') as f:
        for row in tableB:
            name=row[name_col]
            name=normalize_name(name).strip()
            if len(name)==0:
                name="NONE"
            f.write(name+"\n")


# method for adding the translated categories with original data - must point to the DM standard data folder,
# columns will be added to tableA, B
def merge_addTranslation(inFolder):
    print("loading dataset...")
    tableA = pd.read_csv(inFolder + "/tableA.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         )
    header= list(tableA.columns.values)
    header.insert(2,"mtname")

    tableA=tableA.fillna('').to_numpy()
    tableB = pd.read_csv(inFolder + "/tableB.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    linesA=[]
    with open(inFolder+"/nameA.txt.catwords") as f:
        linesA = f.readlines()

    linesB = []
    with open(inFolder + "/nameB.txt.catwords") as f:
        linesB = f.readlines()

    with open(inFolder + "/tableA.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer.writerow(header)

        count=0
        for row in tableA:
            mtn = linesA[count].strip()
            row=list(row)
            row.insert(2, mtn)
            writer.writerow(row)
            count+=1

    with open(inFolder + "/tableB.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer.writerow(header)

        count = 0
        for row in tableB:
            mtn = linesB[count].strip()
            row = list(row)
            row.insert(2, mtn)
            writer.writerow(row)
            count += 1

# input must be a folder (can be recursive) containing the following three files:
# train.csv, test.csv, validation.csv. These must conform to the format required by DM
# these files will be overwritten, by keeping only the name properties. All other properties are deleted, unless
# there are the 'mt_' column
def keep_name_only(in_folder):
    result = list(Path(in_folder).rglob("*.csv"))
    for f in result:
        f=str(f)
        if "train.csv" in f or "test.csv" in f or "validation.csv" in f:
            df = pd.read_csv(f,header=0, encoding="utf-8")
            #findout which columns need to keep
            headers=list(df.columns.values)
            left_n=-1
            right_n=-1
            #left_id=-1
            #right_id=-1
            left_mtname=-1
            right_mtname=-1

            for i in range(0, len(headers)):
                h = headers[i]
                #if (h.lower()=="left_id"):
                #    left_id=i
                #elif (h.lower()=="right_id"):
                #    right_id=i
                if("_name" in h.lower() and "left_" in h.lower() and left_n==-1):
                    left_n=i
                elif ("_title" in h.lower() and "left_" in h.lower() and left_n == -1):
                    left_n = i
                elif ("_name" in h.lower() and "right_" in h.lower() and right_n == -1):
                    right_n = i
                elif ("_title" in h.lower() and "right_" in h.lower() and right_n == -1):
                    right_n = i
                elif ("mtname" in h.lower() and "left_" in h.lower()):
                    left_mtname=i
                elif ("mtname" in h.lower() and "right_" in h.lower()):
                    right_mtname=i

            #start writing the new data
            outf= open(f, 'w', newline='\n', encoding='utf-8')
            print("writing "+f)
            writer = csv.writer(outf, delimiter=",", quoting=csv.QUOTE_ALL)
            #newheader=["id","label","left_id", "left_Name"]
            newheader = ["id", "label", "left_Name"]
            if left_mtname!=-1:
                newheader.append("left_mtname")
            #newheader.append("right_id")
            newheader.append("right_Name")
            if right_mtname!=-1:
                newheader.append("right_mtname")
            writer.writerow(newheader)

            for row in df.values:
                newrow=[row[0], row[1], row[left_n]]
                if left_mtname!=-1:
                    newrow.append(row[left_mtname])
                #newrow.append(row[right_id])
                newrow.append(row[right_n])
                if right_mtname!=-1:
                    newrow.append(row[right_mtname])
                writer.writerow(newrow)


            outf.close()

if __name__ == "__main__":
    #convert data in dm format but keep only names and mtnames if any
    keep_name_only("/home/zz/Work/data/entity_linking/deepmatcher/processed")
    keep_name_only("/home/zz/Work/data/entity_linking/deepmatcher/processed_mt")
    #keep_name_only("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt")
    #keep_name_only("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_original")
    exit(0)

    # #take the downloaded DM datasets containing tableA, B etc, create the required trian/test/valid for DM
    # #WARNING: will overwrite data
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Structured/iTunes-Amazon")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Structured/Amazon-Google")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Structured/Beer")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Structured/Fodors-Zagats")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Structured/Walmart-Amazon")
    #
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Dirty/Walmart-Amazon")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Dirty/iTunes-Amazon")
    #
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Textual/abt_buy_exp_data")
    # parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/Textual/dirty_itunes_amazon_exp_data")


    ##########################################################
    # the following code until another block of #### shows how to
    # take names from dm datasets, pass them to translation
    # to create cat words, then merge them back into the dm
    # datasets, and then create the train/val/test required by dm
    ##########################################################
    # 1. take the names from the downloaded DM datasets (tableA, B),output them as list into the original folder
    #WARNING: will overwrite data
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Structured/iTunes-Amazon",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Structured/Amazon-Google",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Structured/Beer",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Structured/Fodors-Zagats",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Structured/Walmart-Amazon",1)

    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Dirty/Walmart-Amazon",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Dirty/iTunes-Amazon",1)

    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Textual/abt_buy_exp_data",1)
    extract_names("/home/zz/Work/data/entity_linking/deepmatcher/original/Textual/dirty_itunes_amazon_exp_data",1)

    # 2. now upload the folder containing nameA, nameB to IR server and run the openmnt script for translation

    # 3. take the translated name=> catwords and add them to the original dm datasets
    # Warning: will overwrite data
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/iTunes-Amazon")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Amazon-Google")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Beer")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Fodors-Zagats")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Walmart-Amazon")

    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Dirty/Walmart-Amazon")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Dirty/iTunes-Amazon")

    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Textual/abt_buy_exp_data")
    merge_addTranslation("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Textual/dirty_itunes_amazon_exp_data")

    # 4. now run the following code to recreate the train/val/test sets needed by DM
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/iTunes-Amazon")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Amazon-Google")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Beer")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Fodors-Zagats")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Structured/Walmart-Amazon")

    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Dirty/Walmart-Amazon")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Dirty/iTunes-Amazon")

    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Textual/abt_buy_exp_data")
    parse_dm_datasets("/home/zz/Work/data/entity_linking/deepmatcher/original_mt/Textual/dirty_itunes_amazon_exp_data")

    ##############################
    # FINISHED
    ##############################
    exit(0)