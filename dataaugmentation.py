import os
import torch
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Loading datasets
train_df, test_df = pd.read_csv('ekdnam_train.csv'), pd.read_csv('ekdnam_test.csv')

train_df.dropna(inplace = True)
test_df.dropna(inplace = True)

concat_df = pd.concat([train_df, test_df], axis=0)
print("Total dataset size: ", concat_df.shape)
concat_df = concat_df.drop_duplicates()
print("Total dataset size after dropping duplicates: ", concat_df.shape)

# Removing noise
pattern1 = r'\n\.|[\n#]|_' #remove \n., \n, #, _
pattern2 = r'\s+' #repace multiple whitespaces with just one

concat_df['X'] = concat_df['X'].str.replace(pattern1, ' ', regex=True)
concat_df['X'] = concat_df['X'].str.replace(pattern2, ' ', regex=True)
concat_df['X'] = concat_df['X'].str.lower() #lowercasing all the data points

# Dividing into class specific dfs
class_0 = concat_df[concat_df["y"] == 0]
class_1 = concat_df[concat_df["y"] == 1]

print(f"Class 0 shape: {class_0.shape} \t Class 1 shape: {class_1.shape}")

texts = list(class_1["X"].values)
print(f"Number of texts to process: {len(texts)}.")

# Running Contextual Word Embedding mask filling approach
aug = naw.ContextualWordEmbsAug(model_path='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', action="substitute", aug_max = None, aug_p = 0.15, device = device, batch_size = 128)
augmented_text = aug.augment(texts)

dic = {'X': augmented_text, 'y': [1]*len(augmented_text)}
aug_df = pd.DataFrame(dic)
aug_df.to_csv("aug.csv", index = False)
del(aug)
print("Finished augmenting texts")

# Running summarization approach
aug = nas.AbstSummAug(model_path='t5-small', max_length = 100, batch_size = 32, device = device)
summarized_text = aug.augment(texts)

dic = {'X': summarized_text, 'y': [1]*len(summarized_text)}
aug_df = pd.DataFrame(dic)
aug_df.to_csv("summ.csv", index = False)

print("Finished summarizing texts")
