

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
import timeit
start = timeit.default_timer()
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random

with open('/home/fariharahman/active_learning/tram/data/training/bootstrap-training-data.json') as f:
    data_json = json.loads(f.read())

data = pd.DataFrame(
    [
        {'text': row['text'], 'label': row['mappings'][0]['attack_id']}
        for row in data_json['sentences']
        if len(row['mappings']) > 0
    ])


splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data based on the 'label' column
for train_index, test_index in splitter.split(data, data['label']):
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]

# Print the number of rows in each set
print(f"Training set has {len(train_set)} rows {train_set['label'].nunique()}.")
print(f"Test set has {len(test_set)} rows. {test_set['label'].nunique()}")


import transformers
import torch
torch.cuda.empty_cache()

cuda = torch.device('cuda')

model = transformers.BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=data['label'].nunique(),
    output_attentions=False,
    output_hidden_states=False,
)
tokenizer = transformers.BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)


model.train().to(cuda)


from sklearn.preprocessing import OneHotEncoder as OHE

encoder = OHE(sparse_output=False)
encoder.fit(data[['label']])

def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

def load_data(x, y, batch_size=10):
    x_len, y_len = x.shape[0], y.shape[0]
    assert x_len == y_len
    for i in range(0, x_len, batch_size):
        slc = slice(i, i + batch_size)
        yield x[slc].to(cuda), y[slc].to(cuda)

def apply_attention_mask(x):
    return x.ne(tokenizer.pad_token_id).to(int)

from sklearn.model_selection import train_test_split
x_labeled, x_unlabeled = train_test_split(train_set, test_size=.99, stratify=train_set['label'])

print(len(x_labeled))


x_train_labeled = tokenize(x_labeled['text'].tolist())
x_train_unlabeled = tokenize(x_unlabeled['text'].tolist())

y_train_labeled = torch.Tensor(encoder.transform(x_labeled[['label']]))
y_train_unlabeled  = torch.Tensor(encoder.transform(x_unlabeled [['label']]))

from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean

optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 100
batch_size = 100


# Active learning loop
for annotation_round in range(num_annotation_rounds):
    print(f"Annotation Round {annotation_round + 1}")
    
    # Train the model on the currently labeled set of data
    for epoch in range(10):  # Adjust as needed
        epoch_losses = []
        for x, y in tqdm(load_data(x_train_labeled, y_train_labeled, batch_size=10)):
            model.zero_grad()
            out = model(x, attention_mask=apply_attention_mask(x), labels=y)
            epoch_losses.append(out.loss.item())
            out.loss.backward()
            optim.step()
        print(f"Epoch {epoch + 1} loss: {mean(epoch_losses)}")
 
    # Use the trained model to make predictions on the unlabeled set
    confidence_scores = []
    with torch.no_grad():
        model.eval()
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x_batch_unlabeled = x_train_unlabeled[i : i + batch_size].to(cuda)
            out = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled))
            logits = out.logits.to('cpu')
            probs = F.softmax(logits, dim=1)
            # entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
            confidence_scores.extend(probs.max(dim=1).values)

    sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i])
    selected_indices = sorted_indices[:num_samples_to_select]

    # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices]])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices]])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]


for epoch in range(10):  # Adjust as needed
    epoch_losses = []
    for x, y in tqdm(load_data(x_train_labeled, y_train_labeled, batch_size=10)):
        model.zero_grad()
        out = model(x, attention_mask=apply_attention_mask(x), labels=y)
        epoch_losses.append(out.loss.item())
        out.loss.backward()
        optim.step()
    print(f"Epoch {epoch + 1} loss: {mean(epoch_losses)}")

filename = "tc_aug" + str(num_annotation_rounds) + ".pth"



model.eval()

preds = []
batch_size = 20

x_test = tokenize(test_set['text'].tolist())

with torch.no_grad():
    for i in range(0, x_test.shape[0], batch_size):
        x = x_test[i : i + batch_size].to(cuda)
        out = model(x, attention_mask=apply_attention_mask(x))
        preds.extend(out.logits.to('cpu'))

predicted_labels = (
    encoder.inverse_transform(
        F.one_hot(
            torch.vstack(preds).softmax(-1).argmax(-1),
            num_classes=50
        )
        .numpy()
    )
    .reshape(-1)
)

print(predicted_labels)

from sklearn.metrics import precision_recall_fscore_support

# Assuming you have predicted_labels and actual labels
predicted = list(predicted_labels)
actual = test_set['label'].tolist()

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")

end = timeit.default_timer()
print("Execution time in seconds:", end - start)