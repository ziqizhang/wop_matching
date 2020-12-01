'''
This file processes the output of 'run_dm_standard.py' and save results to a csv
'''
import csv, os, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


def save_scores(predictions, gs, setting, digits, outfolder):
    outputPredictions(predictions, gs, setting, outfolder)
    filename = os.path.join(outfolder, "results.csv")
    file = open(filename, "a+")
    file.write(setting)

    file.write("N-fold results:\n")
    labels = unique_labels(gs, predictions)
    target_names = ['%s' % l for l in labels]
    p, r, f1, s = precision_recall_fscore_support(gs, predictions,
                                                  labels=labels)
    acc=accuracy_score(gs, predictions)
    mac_prf_line=prepare_score_string(p,r,f1,s,labels,target_names,digits)

    prf_mac_weighted=precision_recall_fscore_support(gs, predictions,
                                                     average='weighted')
    line = mac_prf_line + "\nmacro avg weighted," + \
           str(prf_mac_weighted[0]) + "," + str(prf_mac_weighted[1]) + "," + \
           str(prf_mac_weighted[2]) + "," + str(prf_mac_weighted[3])

    prf = precision_recall_fscore_support(gs, predictions,
                                          average='micro')
    line=line+"\nmicro avg,"+str(prf[0])+","+str(prf[1])+","+\
         str(prf[2])+","+str(prf[3])
    file.write(line)
    file.write("\naccuracy on this run="+str(acc)+"\n\n")

    file.close()

def outputPredictions(pred, truth, setting, outfolder):
    filename = outfolder+"/predictions-"+setting+".csv"
    file = open(filename, "w")
    for p, t in zip(pred, truth):
        if p==t:
            line=str(p)+","+str(t)+",ok\n"
            file.write(line)
        else:
            line=str(p)+","+str(t)+",wrong\n"
            file.write(line)
    file.close()

def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision, recall, f1, support\n"
    for i, label in enumerate(labels):
        string= string+target_names[i]+","
        for v in (p[i], r[i], f1[i]):
            string = string+"{0:0.{1}f}".format(v, digits)+","
        string = string+"{0}".format(s[i])+"\n"
        #values += ["{0}".format(s[i])]
        #report += fmt % tuple(values)

    #average
    string+="mac avg,"
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        string += "{0:0.{1}f}".format(v, digits)+","
    string += '{0}'.format(np.sum(s))
    return string

def parse_results(inFolder, outCSV):
    f=open(outCSV, 'w', newline='\n', encoding='utf-8')
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=",")
    writer.writerow(["data","P","R","F1"])

    for f in os.listdir(inFolder):
        setting=f
        writer.writerow([setting])

        all_scores=[]
        with open(inFolder + "/"+f) as af:
            lines = af.readlines()

            lookforresult=False
            resultID=None
            scores = []
            for l in lines:
                #
                # if l.startswith("Structured") or l.startswith("Dirty") or l.startswith("Textual"):
                if l.startswith("original") or l.startswith("mtcat"):
                    if resultID is None:
                        resultID=l.strip()
                        scores.append(resultID)
                        continue
                    else:
                        resultID = l.strip()
                        lookforresult = False
                        scores=[resultID]


                if "Training done" in l:
                    lookforresult=True
                    continue

                if "Finished Epoch" in l and lookforresult:
                    parts = l.split("|")
                    pre=parts[6].split(":")[1].strip()
                    re = parts[7].split(":")[1].strip()
                    f1 = parts[5].split(":")[1].strip()
                    scores.append(pre)
                    scores.append(re)
                    scores.append(f1)
                    resultID=None
                    lookforresult=False
                    all_scores.append(scores)
                    scores=[]
                    continue


        for s in all_scores:
            writer.writerow(s)
        writer.writerow(["\n"])

if __name__ == "__main__":
    # inFolder="/home/zz/Work/wop_matching/output/dm/raw"
    # outCSV="/home/zz/Work/wop_matching/output/dm/dm_result.csv"
    # parse_results(inFolder,outCSV)

    inFolder = "/home/zz/Work/wop_matching/output/lspm/raw"
    outCSV = "/home/zz/Work/wop_matching/output/lspm/lspm_result.csv"
    parse_results(inFolder, outCSV)