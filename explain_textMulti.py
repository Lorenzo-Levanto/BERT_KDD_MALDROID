import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from neural_network.historydataset import KDDdataset
from neural_network.sentinelbertMulticlass import BertMultiClassificationHeads
from transformers import BertTokenizer, AutoModel
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
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
    test_dataset = KDDdataset(df_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #model
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-medium")
    num_classes = len(label_encoder.classes_)
    model = BertMultiClassificationHeads(
        bert_model=bert_model,
        layer_sizes=[bert_model.config.hidden_size, 128],
        dropout_probs=[0.2]
    )
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.to(device)
    model.eval()

    print("Modello e dataset caricati")

    list_tokens = []
    list_scores = []

    for ele in test_loader:
        input_ids = ele["input_ids"].to(device)
        attention_mask = ele["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)  
        pred_class = torch.argmax(outputs, dim=1).item()

        token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
        reference_indices = token_reference.generate_reference(max_len, device=device).unsqueeze(0)

        lig = LayerIntegratedGradients(model, model.bert_model.embeddings)
        attributions_ig, delta = lig.attribute(
            input_ids,
            reference_indices,
            additional_forward_args=attention_mask,
            target=pred_class,
            return_convergence_delta=True
        )

        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        tokens = input_ids.cpu().numpy().tolist()

        tokens = [tokenizer.convert_ids_to_tokens(t) for t in tokens[0]]

        list_tokens.append(tokens)
        list_scores.append(attributions)

    #save
    with open('xai_list_score.pickle', 'wb') as handle:
        pickle.dump(list_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('xai_list_tokens.pickle', 'wb') as handle:
        pickle.dump(list_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('END EXPLAIN')

