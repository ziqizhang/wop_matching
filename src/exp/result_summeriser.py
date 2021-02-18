import glob
import csv
import os

def parse_acc(value):
    if len(value)==0:
        return value
    acc=value.split("=")[1].strip()
    if ',' in acc:
        acc=acc[0:acc.index(',')]
    return acc.strip()

#results: expected multi-level dict
#lvl1 keys: gs level
#lvl2 keys: algorithm
def write_header(results:dict, csvwr:csv.writer, values:int):
    row=[""]
    for lvl, algs in results.items():
        for alg in algs.keys():
            header=[lvl+"/"+alg]
            for x in range(0, values-1):
                header.append("")
            row.extend(header)

    csvwr.writerow(row)

#use this for the DNN (cnn,lstm,han) results and fasttext results and ...
#NOT to be used by CML (svm etc because the format is different)
def summarise(infile, outfile):
    print("Summarusing features and levels...")

    print("Creating empty result maps...")
    acc_ = {}
    micro_ = {}
    macro_ = {}
    wmacro_ = {}
    tp_ = {}

    df = open(infile).readlines()

    curr_setting = None
    row_counter = 0
    for row in df:
        row_counter+=1
        row = row.strip()
        if 'results' in row:  # start of a new algorithm, init containers
            curr_setting = row+"_row"+str(row_counter)
        elif 'mac avg' in row:  # macro
            stats = row.split(",")[1:-1]  # -1 because the last number is the support
            macro_[curr_setting]=stats
        elif 'macro avg w' in row:  # weighted macro
            stats = row.split(",")[1:-1]
            wmacro_[curr_setting]=stats
        elif 'micro avg' in row:
            stats = row.split(",")[1:-1]
            micro_[curr_setting]=stats
        elif 'accuracy' in row:  # end of a new algorithm
            acc = parse_acc(row)
            acc_[curr_setting]=acc
        elif row.startswith("1,"):
            stats = row.split(",")[1:-1]
            tp_[curr_setting]=stats
        else:
            continue


    print("Output results...")
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(["","Acc"])

        for setting, results in acc_.items():
            row=setting+","+results
            writer.writerow(row.split(","))

        writer.writerow([""])
        writer.writerow(["","Macro P","R","F1"])
        for setting, results in macro_.items():
            row=results
            row.insert(0,setting)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["", "WMcro P","R","F1"])
        for setting, results in wmacro_.items():
            row = results
            row.insert(0, setting)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["", "Micro P","R","F1"])
        for setting, results in micro_.items():
            row = results
            row.insert(0, setting)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["", "TP P","R","F1"])
        for setting, results in tp_.items():
            row = results
            row.insert(0, setting)
            writer.writerow(row)


if __name__ == "__main__":

    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                   "/home/zz/Work/wop/tmp/desc.csv")
    # summarise("/home/zz/Work/wop/output/cml+dnn_mwpd_val/classifier/scores",
    #                                  "/home/zz/Work/wop/output/cml+dnn_mwpd_val/cml_mwpd_val.csv")

    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                  "/home/zz/Work/wop/output/classifier/dnn_d_X_result.csv")

    # summarise("/home/zz/Work/wop/output/classifier/",
    #               "/home/zz/Work/wop/output/classifier/scores.csv")

    # summarise_cml("/home/zz/Work/wop/output/classifier/scores",
    #               "/home/zz/Work/wop/output/classifier/cml_wdc-missed.csv")

    # input="/home/zz/Work/wop_matching/output/bert_dmdataset_proddesc/output/results.csv"
    # output="/home/zz/Work/wop_matching/output/dm_results.csv"

    #input = "/home/zz/Work/wop_matching/output/bert_dm/output/results.csv"
    #output = "/home/zz/Work/wop_matching/output/bert_dm.csv"
    input = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Beer/results.csv"
    output = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Beer/bert_dm.csv"

    summarise(input, output)


