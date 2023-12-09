# conda env create -f environment.yml

import warnings
warnings.filterwarnings("ignore")

import re
import os
import json
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


"""
Step 0: Argparse
"""
parser = argparse.ArgumentParser(description='MED 277')
parser.add_argument('-m', '--model_name', type=str, choices = ["clinical", "pubmed", "pubmed-fulltext", "biolink", "biobert"], help='which huggingface model to use')
parser.add_argument('-n','--n_epochs', default=10, type=int, help='number of epoch to train')
parser.add_argument('-bs', '--train_bs', default=64, type=int, help='mini-batch size')
parser.add_argument('-tbs', '--test_bs', default=64, type=int, help='mini-batch size')
parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, help='learning rate')
parser.add_argument('-dgpus', action='store_true', help='use multiple gpus?')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

print("Finished parsing arguments.")


"""
Step 1: Preparing data
"""

train_df, test_df = pd.read_csv('ekdnam_train.csv'), pd.read_csv('ekdnam_test.csv')

# Droping nan values
train_df.dropna(inplace = True)
test_df.dropna(inplace = True)

pattern1 = r'\n\.|[\n#]|_' #remove \n., \n, #, _
pattern2 = r'\s+' #repace multiple whitespaces with just one

train_df['X'] = train_df['X'].str.replace(pattern1, '', regex=True)
train_df['X'] = train_df['X'].str.replace(pattern2, ' ', regex=True)
train_df['X'] = train_df['X'].str.lower() #lowercasing all the data points

test_df['X'] = test_df['X'].str.replace(pattern1, '', regex=True)
test_df['X'] = test_df['X'].str.replace(pattern2, ' ', regex=True)
test_df['X'] = test_df['X'].str.lower() #lowercasing all the data points

print("Finished processing data.")


"""
Step 2: Initializing model
"""
if FLAGS.model_name == "clinical":
    model_name = "medicalai/ClinicalBERT"
elif FLAGS.model_name == "biolink":
    model_name = "michiyasunaga/BioLinkBERT-base"
elif FLAGS.model_name == "pubmed":
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
elif FLAGS.model_name == "pubmed-fulltext":
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
elif FLAGS.model_name == "biobert":
    model_name = "dmis-lab/biobert-v1.1"
else:
    raise Exception('Invalid model choice!')
    
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
model.to(device)

if FLAGS.dgpus: # distributed GPU computing
    model = torch.nn.DataParallel(model)

print("Finished initializing model.")


"""
Step 3: Making a dataloader
"""

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length = 512,
            padding='max_length',
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        return item

train_list, test_list = train_df.to_dict('list'), test_df.to_dict('list')
train_dataset = CustomDataset(train_list['X'], train_list['y'], tokenizer)
test_dataset = CustomDataset(test_list['X'], test_list['y'], tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.train_bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.test_bs, shuffle=False)

print("Finished creating dataloaders.\n\n")


"""
Step 4: Training
"""

epochs = FLAGS.n_epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Lists to store metrics
train_loss_list = []
train_acc_list = []
test_acc_list = []
test_classification_list = []
learning_rate_list = []

for epoch in range(epochs):
    model.train()
    train_predictions = []
    train_true_labels = []
    total_loss = 0
    
    # training loop
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        total_loss += loss.item()
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        train_predictions.extend(predictions)
        train_true_labels.extend(labels.cpu().numpy())
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch: {epoch}')
    print(f'Training Loss: {average_loss} \t Accuracy: {train_accuracy}')

    # test loop
    model.eval()
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            test_predictions.extend(predictions)
            test_true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy}\n')
    
    train_loss_list.append(average_loss)
    train_acc_list.append(train_accuracy)
    learning_rate_list.append(scheduler.get_last_lr())
    test_acc_list.append(test_accuracy)
    test_classification_list.append(classification_report(test_true_labels, test_predictions))
    
print("\n\nFinished training.")


"""
Step 5: Storing results
"""
data = vars(FLAGS)
data["train_loss"]: train_loss_list
data["train_acc"]: train_acc_list
data["test_acc"]: test_acc_list
data["test_classification"]: test_classification_list
data["learning_rate"]: learning_rate_list

if not os.path.exists("results"):
    os.makedirs("results")

with open(f'results/{model_name.replace("/", "-" )}.json', 'w') as f:
    json.dump(data, f)
    
print("Finished storing results.")