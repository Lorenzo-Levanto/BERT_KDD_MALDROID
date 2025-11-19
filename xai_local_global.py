import pickle
import pandas as pd

with open("xai_list_tokens.pickle", "rb") as f:
    list_tokens = pickle.load(f)

with open("xai_list_score.pickle", "rb") as f:
    list_scores = pickle.load(f)

#original file with ","
df_test = pd.read_pickle("testsentences.pkl")

filtered_examples = []

for sentence, scores in zip(df_test["text"], list_scores):

    # split with ","
    features = [feat.strip() for feat in sentence.split(",")]

    L = min(len(features), len(scores))

    feature_dict = {features[i]: float(scores[i]) for i in range(L)}

    filtered_examples.append(feature_dict)

#local
df_local = pd.DataFrame(filtered_examples)

#global 
df_global = df_local.mean().sort_values(ascending=False).reset_index()
df_global.columns = ["feature", "global_score"]


#save
df_local.to_csv("local_feature_scores.csv", index=False)
df_global.to_csv("global_feature_scores.csv", index=False)

print("File salvati")
