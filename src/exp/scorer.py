'''
This file processes the output of 'run_dm_standard.py' and save results to a csv
'''
import csv, os

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