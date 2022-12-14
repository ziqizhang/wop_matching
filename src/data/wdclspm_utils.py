'''
code for processing wdclspm datasets
'''
import os, gzip,csv,json,re
import pandas as pd

#this method processes the originally downloaded lspm datasets and convert them into the format required by dm
def parse_wdclspm_datasets(inTrain, inVal, inGS,outFolder):
    for f in os.listdir(inTrain):
        if not f.endswith(".gz"):
            continue

        print("processing:"+f)
        setting = f[0:f.index(".")].split("_")

        setting_id=setting[0]+"_"+setting[2]
        n_outFolder=outFolder + "/"+setting_id
        if not os.path.exists(n_outFolder):
            os.makedirs(n_outFolder)

        ft=open(n_outFolder + "/train.csv", 'w', newline='\n', encoding='utf-8')
        fv=open(n_outFolder + "/validation.csv", 'w', newline='\n', encoding='utf-8')
        fg=open(n_outFolder + "/test.csv", 'w', newline='\n', encoding='utf-8')
        writer_train = csv.writer(ft, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer_valid = csv.writer(fv, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer_test = csv.writer(fg, quoting=csv.QUOTE_MINIMAL, delimiter=",")


        trainFile = inTrain+"/"+setting[0]+"_train_"+setting[2]+".json.gz"
        valFile=inVal+"/"+setting[0]+"_valid_"+setting[2]+".csv"
        gsFile = inGS+"/"+setting[0]+"_gs.json.gz"

        val_ids=[]
        # read validation ids
        with open(valFile,'r') as f:
            for l in f.readlines():
                val_ids.append(l.strip())

        #write train/val files
        has_header=False
        with gzip.open(trainFile, 'r') as f:
            count=0
            for line in f:
                count+=1
                if count%1000==0:
                    print("\t"+str(count))
                data=json.loads(line)

                if not has_header:
                    header=["id","label"]
                    for k in data.keys():
                        if "id_" in k or "_id" in k or k=="label" or "identifier" in k:
                            continue
                        values=k.split("_")

                        if values[1]=="left":
                            header.append("left_"+values[0])
                        else:
                            header.append("right_" + values[0])
                    writer_train.writerow(header)
                    writer_valid.writerow(header)
                    writer_test.writerow(header)
                    has_header=True

                pair_id=data["pair_id"]
                row = [pair_id, data["label"]]
                for k, v in data.items():
                    if "id_" in k or "_id" in k or k == "label" or "identifier" in k:
                        continue
                    if v is None:
                        v=""
                    else:
                        v = str(v)
                    v = re.sub('[^0-9a-zA-Z]+', ' ', v).replace("\s+"," ").strip()
                    row.append(v)
                if pair_id in val_ids: #output to validation set
                    writer_valid.writerow(row)
                else:#output to train set
                    writer_train.writerow(row)

        #write test file
        with gzip.open(gsFile, 'rb') as f:
            for line in f:
                data = json.loads(line)

                pair_id = data["pair_id"]
                row = [pair_id, data["label"]]
                for k, v in data.items():
                    if "id_" in k or "_id" in k or k == "label" or "identifier" in k:
                        continue
                    if v is None:
                        v=""
                    else:
                        v = str(v)
                    v = re.sub('[^0-9a-zA-Z]+', ' ', v).replace("\s+", " ").strip()
                    row.append(v)

                writer_test.writerow(row)

        ft.close()
        fv.close()
        fg.close()

#given a folder containing already convereted dm datasets (from wdc), extract all names
#and save them
def extract_names(inFolder, left_name_col, right_name_col):
    train_data = pd.read_csv(inFolder + "/train.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         ).fillna('').to_numpy()
    valid_data = pd.read_csv(inFolder + "/validation.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                             ).fillna('').to_numpy()
    test_data = pd.read_csv(inFolder + "/test.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                             ).fillna('').to_numpy()

    with open(inFolder+"/names.txt", 'w') as f:
        for row in train_data:
            lname=row[left_name_col]
            rname=row[right_name_col]
            f.write(lname+"\n")
            f.write(rname + "\n")
        for row in valid_data:
            lname=row[left_name_col]
            rname=row[right_name_col]
            f.write(lname+"\n")
            f.write(rname + "\n")
        for row in test_data:
            lname=row[left_name_col]
            rname=row[right_name_col]
            f.write(lname+"\n")
            f.write(rname + "\n")

#given a folder containing already convereted dm datasets (from wdc), the list of all names (from train,test,val),
#and the list of all translated mtwords (names.txt.catwords), insert additional column to all
#data to add left_namemt, right_namemt
#WARNING: will overwrite data
def merge_add_mtcat(inFolder, left_name_col, right_name_col):
    #create name to namecatwords dict
    with open(inFolder+"/names.txt") as f:
        names = f.readlines()

    with open(inFolder + "/names.txt.catwords") as f:
        namemt=f.readlines()

    count=0
    name_2_mt={}
    for n in names:
        mt=namemt[count].strip()
        if mt.startswith("null") and mt.endswith("null"):
            mt=""
        name_2_mt[n.strip()] = mt
        count+=1

    #process training_data, val, and test data
    train_data = pd.read_csv(inFolder + "/train.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                         )
    header = list(train_data.columns.values)
    header.insert(left_name_col+1, "left_mtname")
    header.insert(right_name_col+2, "right_mtname")

    with open(inFolder + "/train.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer.writerow(header)

        train_data=train_data.to_numpy()
        for row in train_data:
            row=list(row)
            lname=row[left_name_col]
            lnamemt = name_2_mt[lname]

            rname = row[right_name_col]
            rnamemt = name_2_mt[rname]

            row.insert(left_name_col+1, lnamemt)
            row.insert(right_name_col+2, rnamemt)
            writer.writerow(row)


    valid_data = pd.read_csv(inFolder + "/validation.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                             ).fillna('').to_numpy()
    with open(inFolder + "/validation.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer.writerow(header)

        for row in valid_data:
            row=list(row)
            lname = row[left_name_col]
            lnamemt = name_2_mt[lname]

            rname = row[right_name_col]
            rnamemt = name_2_mt[rname]

            row.insert(left_name_col+1, lnamemt)
            row.insert(right_name_col+2, rnamemt)
            writer.writerow(row)


    test_data = pd.read_csv(inFolder + "/test.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                            ).fillna('').to_numpy()
    with open(inFolder + "/test.csv", 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
        writer.writerow(header)

        for row in test_data:
            row=list(row)
            lname = row[left_name_col]
            lnamemt = name_2_mt[lname]

            rname = row[right_name_col]
            rnamemt = name_2_mt[rname]

            row.insert(left_name_col+1, lnamemt)
            row.insert(right_name_col+2, rnamemt)
            writer.writerow(row)


if __name__ == "__main__":


    # #take the wdc lspm dataset and convert them into format required by DM
    # parse_wdclspm_datasets("/home/zz/Work/data/wdc-lspc/training-sets",
    #                        "/home/zz/Work/data/wdc-lspc/validation-sets",
    #                        "/home/zz/Work/data/wdc-lspc/gold-standards",
    #                        "/home/zz/Work/data/wdc-lspc/wdclspc")

    # # extract names from all datasets and save a names.txt file within that folder
    # extract_names("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small/all_small",2,9)
    # extract_names("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small/cameras_small", 2, 9)
    # extract_names("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small/computers_small", 2, 9)
    # extract_names("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small/shoes_small", 2, 9)
    # extract_names("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small/watches_small", 2, 9)

    # # merge add mt words
    merge_add_mtcat("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt/all_small",2,9)
    merge_add_mtcat("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt/cameras_small", 2, 9)
    merge_add_mtcat("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt/computers_small", 2, 9)
    merge_add_mtcat("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt/shoes_small", 2, 9)
    merge_add_mtcat("/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_mt/watches_small", 2, 9)
