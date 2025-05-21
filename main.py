from neural_network.sentinelbert import BertBinaryClassificationHeads
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
from sklearn.metrics import confusion_matrix
import os #salvataggio?

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

numeroprova = 100



    
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
            labels = batch["labels"].to(device).float()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs#.logits[:, 1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predicted = (logits > 0.5).float().cpu().numpy()
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
        print(f"F1 score train: {f1:.4f}")

        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs#.logits[:, 1] or ['logits'] for bfsq?
                predicted = (logits > 0.5).float().cpu().numpy()
                val_predictions.extend(predicted)
                val_labels.extend(labels.cpu().numpy())

        val_cm = confusion_matrix(val_labels, val_predictions, labels=[0, 1])
        tn_val, fp_val, fn_val, tp_val = val_cm.ravel()
        precision_val = 0 if tp_val + fp_val == 0 else tp_val / (tp_val + fp_val)
        recall_val = 0 if tp_val + fn_val == 0 else tp_val / (tp_val + fn_val)
        f1_val = 0 if precision_val + recall_val == 0 else 2 * (precision_val * recall_val) / (precision_val + recall_val)
        print(f"F1 score val: {f1_val:.4f}")

        train_confusion_data.append({
            "epoch": epoch + 1,
            "f1-score eval": f1_val,
            "f1-score": f1,
            "true negative (anormal classificate_come_anormal)": tn,
            "false positive (anormal classificate_come_normal)": fp,
            "false negative (normal classificate_come_anormal)": fn,
            "true positive (normal classificate_come_normal)": tp
        })

        #Test phase for matrix
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs#.logits[:, 1]
                predicted = (logits > 0.5).float().cpu().numpy()
                test_predictions.extend(predicted)
                test_labels.extend(labels.cpu().numpy())

        test_cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
        print("Matrice di confusione (Test) [Anormal=0, Normal=1]:")
        print(test_cm)
        tn_test, fp_test, fn_test, tp_test = test_cm.ravel()
        precision_test = 0 if tp_test + fp_test == 0 else tp_test / (tp_test + fp_test)
        recall_test = 0 if tp_test + fn_test == 0 else tp_test / (tp_test + fn_test)
        f1_test = 0 if precision_test + recall_test == 0 else 2 * (precision_test * recall_test) / (precision_test + recall_test)
        print(f"F1 score test: {f1_test:.4f}")

        test_confusion_data.append({
            "f1 validation phase": f1_val,
            "epoch": epoch + 1,
            "f1-score": f1_test,
            "true negative (anormal classificate_come_anormal)": tn_test,
            "false positive (anormal classificate_come_normal)": fp_test,
            "false negative (normal classificate_come_anormal)": fn_test,
            "true positive (normal classificate_come_normal)": tp_test
        })
        '''
        #salvataggio con bertforsequenceclassification!
        if f1_val > best_f_score:
            best_f_score = f1_val
            best_model_epoch = epoch + 1
            patience_counter = 0
            try:
                model.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                tokenizer.save_pretrained(f"{model_name}_epoch_{epoch + 1}")
                print(f"Modello salvato: {model_name}_epoch_{epoch + 1}")
            except OSError as e:  # More specific exception
                print(f"Errore durante il salvataggio del modello: {e}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        '''
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

    trainfile = "KDDTrain+aggregateOneCls.csv"
    testfile = "KDDTest+aggregateOneCls.csv"
    columnmapping_file = "columnmapping.json"
    
    #train_sentences_file, test_sentences_file = load_processdata(trainfile, testfile, columnmapping_file)
    train_sentences_file, test_sentences_file = load_processdataPICKLE(trainfile, testfile, columnmapping_file)


    #load for train
    #df_train = pd.read_csv(train_sentences_file).head(numeroprova)
    df_train = pd.read_pickle(train_sentences_file).head(numeroprova)
    
    #label phase for train
    label_encoder = LabelEncoder()
    df_train["label"] = label_encoder.fit_transform(df_train["label"])

    #divide train
    traindf, valdf = train_test_split(df_train, test_size=0.20, stratify=df_train["label"], random_state=42)

    #load for test
    #df_test = pd.read_csv(test_sentences_file).head(numeroprova)
    df_test = pd.read_pickle(test_sentences_file).head(numeroprova)

    #label phase for test
    df_test["label"] = label_encoder.transform(df_test["label"])

    #token phase
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-medium")

    #creation dataset
    train_dataset = KDDdataset(traindf, tokenizer)
    val_dataset = KDDdataset(valdf, tokenizer)
    test_dataset = KDDdataset(df_test, tokenizer)
    
    '''
    #save , puo servire per colab???
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(val_dataset, "val_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")
    tokenizer.save_pretrained("saved_tokenizer")
    print("dataset tokenizzati salvati")
    '''

    #dataloader from torch
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#model
    
    bert_model = AutoModel.from_pretrained('prajjwal1/bert-medium')
    model = BertBinaryClassificationHeads(
        bert_model=bert_model,
        layer_sizes=[bert_model.config.hidden_size, 128],
        dropout_probs=[0.2]
    )
    
    #if want to use bertforsequenceclassification
    #model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-medium", num_labels=2)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    #criterion = nn.BCELoss()

    #optimizer and criterion 
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()  #crossEntropyLoss per logits e argmax

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
            os.makedirs("savemodel", exist_ok=True)
            torch.save(model.state_dict(), "savemodel/model.pt")
            tokenizer.save_pretrained("savemodel")
            print("Modello salvato come savemodel")
        except Exception as e:
            print(f"Errore durante il salvataggio finale del modello: {e}")    
    '''
    #save if use bertfor....
    finally:
        try:
            model.save_pretrained("savemodel")
            tokenizer.save_pretrained("savemodel")
            print("Modello salvato come savemodel")
        except Exception as e:
            print(f"Errore durante il salvataggio finale del modello: {e}")
    '''

