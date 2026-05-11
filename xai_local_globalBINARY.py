import pickle
import pandas as pd

with open("xai_list_tokens.pickle", "rb") as f:
    list_tokens = pickle.load(f)

with open("xai_list_score.pickle", "rb") as f:
    list_scores = pickle.load(f)


filtered_examples = []


for tokens, scores in zip(list_tokens, list_scores):
    
    feature_dict = {}
    
    current_feature = []
    current_scores = []
    
    for token, score in zip(tokens, scores):
        
        #special token check
        if token in ["[CLS]", "[SEP]"]:
            continue
        
            
        #splitting with ","
        if token == ",":
            if not current_feature: continue 

            #reconstruction if find "##"
            reconstructed_text = ""
            for t in current_feature:
                if t.startswith("##"):
                    # if token start with ##, attach with the precedent 
                    reconstructed_text += t[2:] 
                else:
                    #or add space if need 
                    if reconstructed_text: reconstructed_text += " "
                    reconstructed_text += t
            
            #estraction name feature
            if " is " in reconstructed_text:
                feature_name = reconstructed_text.split(" is ")[0].strip()
            else:
                feature_name = reconstructed_text
            

            # score final
            total_score = sum(current_scores)
            feature_dict[feature_name] = total_score

            #reset
            current_feature = []
            current_scores = []
            
        else:
            #add
            current_feature.append(token)
            current_scores.append(score)
            
    #if for last feature without "," for ending
    if current_feature:
        reconstructed_text = ""
        for t in current_feature:
            if t.startswith("##"):
                # if token start with ##, attach with the precedent 
                reconstructed_text += t[2:]
            else:
                #or add space if need
                if reconstructed_text: reconstructed_text += " "
                reconstructed_text += t
        
        #estraction name feature
        if " is " in reconstructed_text:
            feature_name = reconstructed_text.split(" is ")[0].strip()
        else:
            feature_name = reconstructed_text

        if feature_name.endswith(" is"):
            feature_name = feature_name[:-3].strip()#to eliminate " is" at the last feature   
        total_score = sum(current_scores)
        feature_dict[feature_name] = total_score

    filtered_examples.append(feature_dict)

#local
df_local = pd.DataFrame(filtered_examples)

df_local['label_classe'] = list_labels

df_local_0 = df_local[df_local['label_classe'] == 0].drop(columns=['label_classe'])
df_local_1 = df_local[df_local['label_classe'] == 1].drop(columns=['label_classe'])


df_global_0 = df_local_0.mean().sort_values(ascending=False).reset_index()
df_global_0.columns = ["feature", "global_mean_score"]

df_global_1 = df_local_1.mean().sort_values(ascending=False).reset_index()
df_global_1.columns = ["feature", "global_mean_score"]



df_local.drop(columns=['label_classe']).to_csv("local_feature_scores.csv", index=False)

df_global_0.to_csv("global_feature_scores_class0.csv", index=False)
df_global_1.to_csv("global_feature_scores_class1.csv", index=False)

print("Done")


