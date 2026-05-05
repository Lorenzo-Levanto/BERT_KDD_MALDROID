#anormal =0 anormal =1
from neural_network.sentinelbertMulticlass import BertMultiClassificationHeads
from neural_network.historydataset import KDDdataset
from preprocessing.load_create_convert import load_processdataPICKLE, load_processdata
import pandas as pd
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import os 

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def fine_tuning(model, optimizer, train_dataloader, val_dataloader, test_dataloader, criterion, patience, epochs, best_f_score, model_name, device):
    patience_counter = 0
    best_model_epoch = 0
    train_confusion_data = []
    test_confusion_data = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_predictions = []
        train_labels = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            train_predictions.extend(predicted)
            train_labels.extend(labels.cpu().numpy())

        train_cm = confusion_matrix(train_labels, train_predictions, labels=[0, 1, 2, 3, 4])
        print(f"\nepoca {epoch + 1} completata")
        print("Matrice di confusione (Train):")
        print(train_cm)
        
        f1 = f1_score(train_labels, train_predictions, average="macro")

        print(f"F1 score train: {f1:.4f}")

        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).long()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                val_predictions.extend(predicted)
                val_labels.extend(labels.cpu().numpy())

        val_cm = confusion_matrix(val_labels, val_predictions, labels=[0, 1, 2, 3, 4])
        f1_val = f1_score(train_labels, train_predictions, average="macro")

        print(f"F1 score val: {f1_val:.4f}")

        
        train_confusion_data.append({
            "epoch": epoch + 1,
            "f1-score eval": f1_val,
            "f1-score train": f1,
            "matrix train": train_cm
        })
        
        #Test phase
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).long()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                test_predictions.extend(predicted)
                test_labels.extend(labels.cpu().numpy())

        test_cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2, 3, 4])
        print("matrice di confusione (Test):")
        print(test_cm)
        f1_test = f1_score(test_labels, test_predictions, average="macro")

        print(f"F1 score test: {f1_test:.4f}")
        
        test_confusion_data.append({
            "f1 validation phase": f1_val,
            "epoch": epoch + 1,
            "f1-score test": f1_test,
            "matrix ": test_cm
        })
        
        #save
        if f1_val > best_f_score:
            best_f_score = f1_val
            best_model_epoch = epoch + 1
            patience_counter = 0
            try:
                save_dir = f"{model_name}_epoch_{epoch + 1}"
                os.makedirs(save_dir, exist_ok=True)
                
                torch.save({
                    'bert_model_state_dict': model.bert_model.state_dict(),
                    'classifier_state_dict': model.state_dict(),
                }, os.path.join(save_dir, "model.pth"))
                
                tokenizer.save_pretrained(save_dir)
                print(f"Modello salvato: {save_dir}")
            except OSError as e:
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
    #set seed
    seed = 42
    set_seed(seed)

    trainfile = "train_scaler_70_percent_seed_70.csv"
    testfile = "test_scaler_30_percent_seed_70.csv"
    columnmapping_file = "columnmapping_CICMaldroid.json"
    
    #train_sentences_file, test_sentences_file = load_processdata(trainfile, testfile, columnmapping_file)
    train_sentences_file, test_sentences_file = load_processdataPICKLE(trainfile, testfile, columnmapping_file)

    
    #load for train
    #df_train = pd.read_csv(train_sentences_file)
    df_train = pd.read_pickle(train_sentences_file)
    
    

    #label phase for train
    label_encoder = LabelEncoder()
    df_train["label"] = label_encoder.fit_transform(df_train["label"])

    #divide train
    traindf, valdf = train_test_split(df_train, test_size=0.20, stratify=df_train["label"], random_state=42)

    #load for test
    #df_test = pd.read_csv(test_sentences_file)
    df_test = pd.read_pickle(test_sentences_file)


    #label phase for test
    df_test["label"] = label_encoder.transform(df_test["label"])

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
    bert_model = AutoModel.from_pretrained('prajjwal1/bert-medium')
    model = BertMultiClassificationHeads(
        bert_model=bert_model,
        layer_sizes=[bert_model.config.hidden_size, 512],
        dropout_probs=[0.2]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #optimizer and criterion 
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()  

    #fine tuning phase
    try:    
        fine_tuning(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            criterion,
            patience=3,
            epochs=15,
            best_f_score=-float("inf"),
            model_name="savemodel",
            device=device
        )
    except Exception as e:
        print(f"Errore durante il fine-tuning: {e}")
    finally:
        try:
            os.makedirs("savemodel", exist_ok=True)
            torch.save(model.state_dict(), "savemodel/model.pt")
            tokenizer.save_pretrained("savemodel")
            print("Modello salvato come savemodel")
        except Exception as e:
            print(f"Errore durante il salvataggio finale del modello: {e}")    



