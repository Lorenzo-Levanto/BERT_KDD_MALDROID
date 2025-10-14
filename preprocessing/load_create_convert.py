import pandas as pd
import json
import pickle

#function
def convertrowtosentence(row):
    try:
        values = [f"{col} is {row[col]}" for col in row.index[:-1] if pd.notna(row[col])]
        return ", ".join(values)
    except Exception as e:
        print(f"Errore nella conversione della riga {row.name}: {e}")
        return "invalid_sentence"
    
def load_processdata(trainfile, testfile, columnmapping_file):
        
        with open("columnmapping.json", "r") as f:
            columnmapping = json.load(f)
        
        
        #load and create trainsentences
        df_train = pd.read_csv(trainfile, encoding="latin1")
        df_train.columns = df_train.columns.str.replace('Ã¬', 'i', regex=False).str.strip()
        df_train.rename(columns=columnmapping, inplace=True)
        df_train["text"] = df_train.apply(convertrowtosentence, axis=1)
        df_train = df_train[["text", "label"]]
        df_train.to_csv("trainsentences.csv", index=False)
        print("File trainsentences creato")
        
        #load and create testsentences
        df_test = pd.read_csv(testfile, encoding="latin1")
        df_test.columns = df_test.columns.str.replace('Ã¬', 'i', regex=False).str.strip()
        df_test.rename(columns=columnmapping, inplace=True)
        df_test["text"] = df_test.apply(convertrowtosentence, axis=1)
        df_test = df_test[["text", "label"]]
        df_test.to_csv("testsentences.csv", index=False)
        print("File testsentences creato")
        
        return "trainsentences.csv", "testsentences.csv"
    
def load_processdataPICKLE(trainfile, testfile, columnmapping_file):
        
        with open("columnmapping.json", "r") as f:
            columnmapping = json.load(f)
        
        
        #load and create trainsentences
        df_train = pd.read_csv(trainfile, encoding="latin1")
        df_train.columns = df_train.columns.str.replace('Ã¬', 'i', regex=False).str.strip()
        df_train.rename(columns=columnmapping, inplace=True)
        df_train["text"] = df_train.apply(convertrowtosentence, axis=1)
        df_train = df_train[["text", "label"]]
        df_train.to_pickle("trainsentences.pkl")
        print("File trainsentences creato")
        
        #load and create testsentences
        df_test = pd.read_csv(testfile, encoding="latin1")
        df_test.columns = df_test.columns.str.replace('Ã¬', 'i', regex=False).str.strip()
        df_test.rename(columns=columnmapping, inplace=True)
        df_test["text"] = df_test.apply(convertrowtosentence, axis=1)
        df_test = df_test[["text", "label"]]
        df_test.to_pickle("testsentences.pkl")
        print("File testsentences creato")
        

        return "trainsentences.pkl", "testsentences.pkl"
