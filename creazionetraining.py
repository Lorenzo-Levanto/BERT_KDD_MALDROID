import pandas as pd
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

numeroprova = 100

#function
def convertrowtosentence(row):
    try:
        #uso le colonne presenti nel DataFrame (escludi l'ultima colonna, che è "label")
        values = [f"{col} is {row[col]}" for col in row.index[:-1] if pd.notna(row[col])]
        return ", ".join(values)
    except Exception as e:
        print(f"Errore nella conversione della riga {row.name}: {e}")
        return "invalid_sentence"



class KDDdataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)  # Manteniamo float per BCELoss
        }

def fine_tuning(model, optimizer, train_dataloader, val_dataloader, test_dataloader, criterion, patience, epochs, best_f_score, model_name, device):
    patience_counter = 0
    best_model_epoch = 0
    train_confusion_data = []
    test_confusion_data = []  # Aggiunto per salvare i dati del test set

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1]
            probs = torch.sigmoid(logits)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (probs > 0.5).float().cpu().numpy()
            train_predictions.extend(predicted)
            train_labels.extend(labels.cpu().numpy())

        train_cm = confusion_matrix(train_labels, train_predictions, labels=[0, 1])
        print(f"\nEpoca {epoch + 1} completata")
        print("Matrice di confusione (Train) [Anormal=0, Normal=1]:")
        
        tn, fp, fn, tp = train_cm.ravel()
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
        

        train_confusion_data.append({
            "epoch": epoch + 1,
            "f1-score": f1,
            "true negative (anormal classificate_come_anormal)": tn,
            "false positive (anormal classificate_come_normal)": fp,
            "false negative (normal classificate_come_anormal)": fn,
            "true positive (normal classificate_come_normal)": tp
        })

        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, 1]
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float().cpu().numpy()
                val_predictions.extend(predicted)
                val_labels.extend(labels.cpu().numpy())
        
        val_cm = confusion_matrix(val_labels, val_predictions, labels=[0, 1])
        #print("Matrice di confusione (Validation) [Anormal=0, Normal=1]:")
        #print(val_cm)
        tn_val, fp_val, fn_val, tp_val = val_cm.ravel()
        precision_val = 0 if tp_val + fp_val == 0 else tp_val / (tp_val + fp_val)
        recall_val = 0 if tp_val + fn_val == 0 else tp_val / (tp_val + fn_val)
        f1_val = 0 if precision_val + recall_val == 0 else 2 * (precision_val * recall_val) / (precision_val + recall_val)
        

        #Test phase for matrix
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)  # Non passare labels
                logits = outputs.logits[:, 1]
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float().cpu().numpy()
                test_predictions.extend(predicted)
                test_labels.extend(labels.cpu().numpy())

        test_cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
        tn_test, fp_test, fn_test, tp_test = test_cm.ravel()
        precision_test = 0 if tp_test + fp_test == 0 else tp_test / (tp_test + fp_test)
        recall_test = 0 if tp_test + fn_test == 0 else tp_test / (tp_test + fn_test)
        f1_test = 0 if precision_test + recall_test == 0 else 2 * (precision_test * recall_test) / (precision_test + recall_test)

        test_confusion_data.append({
            "epoch": epoch + 1,
            "f1_score": f1_test,
            "true negative (anormal classificate_come_anormal)": tn_test,
            "false positive (anormal classificate_come_normal)": fp_test,
            "false negative (normal classificate_come_anormal)": fn_test,
            "true positive (normal classificate_come_normal)": tp_test
        })
        


        if f1_val > best_f_score:
            best_f_score = f1_val
            best_model_epoch = epoch + 1
            patience_counter = 0
            try:
                model.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                tokenizer.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                print(f"Modello salvato: {model_name}_epoch_{epoch + 1}")
            except Exception as e:
                print(f"Errore durante il salvataggio del modello: {e}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Miglior modello salvato dall'epoca {best_model_epoch}")

    train_confusion_df = pd.DataFrame(train_confusion_data)
    train_confusion_df.to_csv("matricetrain.csv", index=False)
    print("matrice di confusione del file train salvata come matricetrain.csv")

    test_confusion_df = pd.DataFrame(test_confusion_data)
    test_confusion_df.to_csv("matricetest.csv", index=False)
    print("matrice di confusione del file test salvata come matricetest.csv")

if __name__ == '__main__':
    trainfile = "KDDTrain+aggregateOneCls.csv"
    testfile = "KDDTest+aggregateOneCls.csv"

    #load and create trainsentences
    df_train = pd.read_csv(trainfile)
    df_train["text"] = df_train.apply(convertrowtosentence, axis=1)
    df_train = df_train[["text", " classification."]]
    df_train=df_train.rename(columns={" classification.":"label"})
    df_train.to_csv("trainsentences.csv", index=False)
    print("File trainsentences creato")

    #load for train
    df_train = pd.read_csv("trainsentences.csv").head(numeroprova)

    #label phase for train
    labele = LabelEncoder()
    df_train["label"] = labele.fit_transform(df_train["label"])  # Anormal=0, Normal=1

    #divide train
    traindf, valdf = train_test_split(df_train, test_size=0.15, random_state=42)

    #load and create testsentences
    df_test = pd.read_csv(testfile)
    df_test["text"] = df_test.apply(convertrowtosentence, axis=1)
    df_test = df_test[["text", " classification."]]
    df_test=df_test.rename(columns={" classification.":"label"})
    df_test.to_csv("testsentences.csv", index=False)
    print("File testsentences creato")

    #load for test
    df_test = pd.read_csv("testsentences.csv").head(numeroprova)

    #Uniforma le etichette (converti tutto in minuscolo)
    #df_test["label"] = df_test["label"].str.lower()
    #print("Etichette nel file di test dopo uniformazione:")
    #print(df_test["label"].head())
    #print(df_test["label"].dtype)

    #label phase for test
    df_test["label"] = labele.fit_transform(df_test["label"])

    #token phase
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-medium")

    #creation dataset
    train_dataset = KDDdataset(traindf, tokenizer)
    val_dataset = KDDdataset(valdf, tokenizer)
    test_dataset = KDDdataset(df_test, tokenizer)

    #dataloader from torch
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    #model
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-medium", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    #fine tuning phase
    try:
        fine_tuning(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            criterion,
            patience=10,
            epochs=10,
            best_f_score=-float("inf"),
            model_name="savemodel",
            device=device
        )
    except Exception as e:
        print(f"Errore durante il fine-tuning: {e}")
    finally:
        try:
            model.save_pretrained("savemodel")
            tokenizer.save_pretrained("savemodel")
            print("Completato e modello salvato come savemodel")
        except Exception as e:
            print(f"Errore durante il salvataggio finale del modello: {e}")
