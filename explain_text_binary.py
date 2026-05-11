import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from neural_network.historydataset import KDDdataset
from neural_network.sentinelbert import BertBinaryClassificationHeads
from transformers import AutoModel, AutoTokenizer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
import pandas as pd
from transformers import BertTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device-->', device)

    
    test_sentences_file = "testsentences.pkl"  
    model_dir = "savemodel"  
    max_len = 512

    #dataset
    df_test = pd.read_pickle(test_sentences_file)



    label_encoder = LabelEncoder()
    df_test["label"] = label_encoder.fit_transform(df_test["label"])

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    test_dataset = KDDdataset(df_test, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #model
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-medium")
    num_classes = len(label_encoder.classes_)
    model = BertBinaryClassificationHeads(
        bert_model=bert_model,
        layer_sizes=[512, 128],
        dropout_probs=[0.2]
    )
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.to(device)
    model.eval()
    model.zero_grad()

    print("Modello e dataset caricati")

    #vis_data_records = []
    list_tokens = []
    list_scores = []
    list_labels = []

    print('START EXPLAIN')
    for ele in test_loader:
        input_ids = ele['input_ids'].to(device)
        attention_mask = ele['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        label = ele['labels'].item()

        pred = output.item()
        pred_ind  = (output > 0.5).float().item()
        if pred_ind == label:
            token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
            reference_indices = token_reference.generate_reference(512, device=device).unsqueeze(0)

            lig = LayerIntegratedGradients(model, model.bert_model.embeddings)
            attributions_ig, delta = lig.attribute(input_ids, reference_indices, additional_forward_args=attention_mask, \
                                                    return_convergence_delta=True)

            attributions = attributions_ig.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.cpu().detach().numpy()
            tokens = input_ids.cpu().numpy().tolist()

            tokens = [tokenizer.convert_ids_to_tokens(t) for t in tokens[0]]
            list_tokens.append(tokens)
            list_scores.append(attributions)
            list_labels.append(label)



    with open('xai_list_score.pickle', 'wb') as handle:
        pickle.dump(list_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('xai_list_tokens.pickle', 'wb') as handle:
        pickle.dump(list_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('xai_list_labels.pickle', 'wb') as handle:
        pickle.dump(list_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('END EXPLAIN')
