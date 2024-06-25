import pandas as pd
import fasttext 
import os
import re

WORKDIR = os.getcwd()

def clean_text (sent : str) -> str :
    sent = sent.lower() 
    patterns = [r'\((.*)\)', 
                r'\<(.*)\>', 
                r'[\\|\+|\=|\-|\_|\*|\/]',
                r'[0-9]+'
                ]
    for pattern in patterns :
       sent = re.sub(pattern,r"", sent) 
    return sent

def get_corpus (data : str) -> str:
    dataset = pd.read_csv(data) 
    dataset.dropna(inplace=True,axis=0) 
    sentences = dataset["text"].apply(lambda x : clean_text(x) ).tolist() 
    return "".join(sentences)

if __name__ == "__main__":
    dataPath = os.path.join(WORKDIR, "Train.csv")
    corpus = get_corpus(dataPath)
    
    with open (os.path.join(WORKDIR,"train-corpus.txt"), "w") as file: 
        file.write(corpus)

    model = fasttext.train_unsupervised(
            input = os.path.join(WORKDIR,"train-corpus.txt"),
            dim = 32,
            epoch = 20,
            minCount = 1,
            )
    model.save_model("ft-model.bin")
    os.remove(os.path.join(WORKDIR, "train-corpus.txt"))
