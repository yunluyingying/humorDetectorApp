import pandas as pd
import json

def getData(filePath):
    """
    Input is the input data file path
    Returns a pandas dataframe
    """
    inputFile = open(filePath,'r')
    text = []
    label = []
    for line in inputFile:
            data = json.loads(line)
            funny = data["funny"]
            if funny >= 20:
                text.append(data["text"])
                label.append(20)
            elif funny >= 2:
                text.append(data["text"])
                label.append(funny)
    # get normalized scoring
    label_normalized = []
    min_value = min(label)
    max_value = max(label)
    for value in label:
        tmp = round((value - min_value) * 10/(max_value - min_value))
        label_normalized.append(tmp)
    df = pd.DataFrame({"text": text, "score": label_normalized})

    return df

if __name__ == "__main__":

    fileInPath = "../../inputData/review.json"
    data = getData(fileInPath)
    # save data to csv file
    #print(data["score"])
    data.to_csv("../../outputData/train_data.csv",index=None)
    print("------------------- Data saved in CSV------------------------")
