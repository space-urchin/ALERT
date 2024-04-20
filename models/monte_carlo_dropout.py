
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
import timeit
start = timeit.default_timer()

nltk.download('punkt')


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

print(len(train_index), len(test_index))

# Print the number of rows in each set
print(f"Training set has {len(train_set)} rows {train_set['label'].nunique()}.")
print(f"Test set has {len(test_set)} rows. {test_set['label'].nunique()}")


import transformers
import torch

cuda = torch.device('cuda')

model = transformers.BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=data['label'].nunique(),
    output_attentions=False,
    output_hidden_states=True,
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
num_annotation_rounds = 50
batch_size = 100

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()
        self.dropout = nn.Dropout(0.9)
        self.fc = nn.Linear(input_size, num_classes)  # Adjust output size for your task

    def forward(self, x):
        x = self.dropout(x)
        return torch.softmax(self.fc(x), dim=1)  # Softmax activation for multi-class classification


def get_model(input_shape, num_classes):
    model = Perceptron(input_shape, num_classes)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    return model, optimizer

p_model, optimizer  = get_model(768, 50)

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

    # if (annotation_round + 1) == 50 or (annotation_round + 1) == 100 or (annotation_round + 1) == 150:
    #     filename = "coreset_1" + str(annotation_round) + ".pth"
    #     torch.save(model.state_dict(), filename)

        
    cls_vector = []
    with torch.no_grad():
        for x, y in tqdm(load_data(x_train_labeled, y_train_labeled, batch_size=10)):
            outputs = model(x, attention_mask=apply_attention_mask(x))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector = torch.stack(cls_vector)
    print(cls_vector.shape)


    for epoch in range(10):
        epoch_losses = []
        for i in range(0, cls_vector.shape[0], 10):
            p_model.train().to('cpu')
            x = cls_vector[i : i + 10]
            y = y_train_labeled[i : i + 10]
            optimizer.zero_grad()
            outputs = p_model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)  
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())  # Append the loss of the current batch

    print(f'Losses at epoch {epoch}: {epoch_losses}')  # Print the losses after each epoch


    confidence_scores = []
    cls_vector_eval = []
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(load_data(x_train_unlabeled, y_train_unlabeled, batch_size=10)):
            model.zero_grad()
            outputs = model(x, attention_mask=apply_attention_mask(x))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector_eval.extend(last_hidden_states[:, 0, :].to('cpu'))

    num_inference_cycles = 10

    entropies = []
    cls_vector_eval = torch.stack(cls_vector_eval)

    # Repeat inference 10 times
    for _ in range(num_inference_cycles):
        preds=[]
        with torch.no_grad():
            p_model.eval().to(cuda)
            p_model.dropout.train()
            for i in range(0, cls_vector_eval.shape[0], 10):
                x_batch_unlabeled = cls_vector_eval[i : i + 10].to(cuda)
                out = p_model(x_batch_unlabeled)
                preds.extend(out.to('cpu'))
        preds = torch.stack(preds)
        entropy = -torch.sum(preds * torch.log(preds), dim=1)  # Entropy formula
        entropies.append(entropy)

    print(len(entropies), entropies[0].shape)
    average_entropy = torch.mean(torch.stack(entropies), dim=0)
    print(average_entropy.shape)

    # Sort indices based on maximum entropy
    sorted_indices = torch.argsort(average_entropy, descending=True)

    selected_indices = sorted_indices[:num_samples_to_select]
    print(selected_indices)

    # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices]])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices]])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]

    if (annotation_round + 1) == num_annotation_rounds:
        for epoch in range(10):  # Adjust as needed
            epoch_losses = []
            for x, y in tqdm(load_data(x_train_labeled, y_train_labeled, batch_size=10)):
                model.zero_grad()
                out = model(x, attention_mask=apply_attention_mask(x), labels=y)
                epoch_losses.append(out.loss.item())
                out.loss.backward()
                optim.step()
            print(f"Epoch {epoch + 1} loss: {mean(epoch_losses)}")
        filename = "mcdtime_1" + str(annotation_round) + ".pth"
        torch.save(model.state_dict(), filename)






# torch.save(model.state_dict(), filename)




import torch.nn.functional as F


print(len(x_train_labeled))

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

print("mcd")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")

end = timeit.default_timer()
print("Execution time in seconds:", end - start)

