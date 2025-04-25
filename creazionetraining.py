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

#set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

numeroprova=100

#function
def convertrowtosentence(row):
    try:
        values = [f"{col} is {row[col]}" for col in columnnames()[:-1] if pd.notna(row[col])]
        return ", ".join(values)
    except Exception as e:
        print(f"Errore nella conversione della riga {row.name}: {e}")
        return "invalid_sentence"

def columnnames():
    return [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label"
    ]

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
            "labels": torch.tensor(label, dtype=torch.float)
        }

def fine_tuning(model, optimizer, train_dataloader, val_dataloader, criterion, patience, epochs, best_f_score, model_name, device):
    patience_counter = 0
    best_model_epoch = 0  # Tieni traccia dell'epoca con il miglior F1-score
    
    train_confusion_data = []
    
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
        print(train_cm)

        tn, fp, fn, tp = train_cm.ravel()
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
        print(f" F1 score val: {f1:.4f}")
        
        train_confusion_data.append({
            "epoch": epoch + 1,
            "f1_score": f1,
            #"matrix": train_cm,
            "true negative(anormal classificate come anormal)": tn,
            "false positive(anormal classificate come normal)": fp,
            "false negative(normal classificate come anormal)": fn,
            "true positive(normal classificate come normal)": tp
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
                loss = criterion(probs, labels)
                predicted = (probs > 0.5).float().cpu().numpy()
                val_predictions.extend(predicted)
                val_labels.extend(labels.cpu().numpy())
        #withtoch.....

        val_cm = confusion_matrix(val_labels, val_predictions, labels=[0, 1])
        print("Matrice di confusione (Validation) [Anormal=0, Normal=1]:")
        print(val_cm)

        tn,fp, fn, tp = val_cm.ravel()
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
        print(f" F1 score val: {f1:.4f}")

        if f1 > best_f_score:
            best_f_score = f1
            best_model_epoch = epoch + 1
            patience_counter = 0
            #save model as savemodel_epoch_X
            try:
                model.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                tokenizer.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                print(f"Modello salvato: {model_name}_epoch_{epoch + 1}")
                # Salva anche come savemodel (il migliore finora)
                #model.save_pretrained("savemodel")
                #tokenizer.save_pretrained("savemodel")
                #print(f"Modello migliore aggiornato: {model_name}")
            except Exception as e:
                print(f"Errore durante il salvataggio del modello: {e}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Miglior modello salvato dall'epoca {best_model_epoch} con F1-score: {best_f_score:.4f}")
    
    train_confusion_df = pd.DataFrame(train_confusion_data)
    train_confusion_df.to_csv("matricetrain.csv", index=False)
    print(f"matrice di confusione del file train salvata come matricetest.csv")
    
    

if __name__ == '__main__':
    
    
    trainfile = "KDDTrain+aggregateOneCls.csv"

    #load and create traisenteces
    df_train = pd.read_csv(trainfile, header=0, names=columnnames())
    df_train["text"] = df_train.apply(convertrowtosentence, axis=1)
    df_train = df_train[["text", "label"]]
    df_train.to_csv("trainsentences.csv", index=False)
    print("File trainsentences creato")

    #load numeroprova for train
    df_train = pd.read_csv("trainsentences.csv").head(numeroprova)
 

    #label phase
    labele = LabelEncoder()
    df_train["label"] = labele.fit_transform(df_train["label"])  # Anormal=0, Normal=1

    #divide train
    traindf, valdf = train_test_split(df_train, test_size=0.15, random_state=42)

    #token phase
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-medium")

    #creation dataset
    train_dataset = KDDdataset(traindf, tokenizer)
    val_dataset = KDDdataset(valdf, tokenizer)

    # dataloader from torch
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #model
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-medium", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    #fine tuning phase
    try:
        fine_tuning(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            criterion,
            patience=10,
            epochs=10,
            best_f_score=-float("inf"),
            model_name="savemodel",
            device=device
        )
    except Exception as e:
        print(f"errore durante il fine-tuning: {e}")
    finally:
        # save final model
        try:
            model.save_pretrained("savemodel")
            tokenizer.save_pretrained("savemodel")
            print("completato e modello salvato come savemodel")
        except Exception as e:
            print(f"errore durante il salvataggio finale del modello: {e}")

