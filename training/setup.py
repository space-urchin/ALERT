import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder as OHE
import transformers
import torch

# Load data
with open('../data/training-data.json') as f:
    data_json = json.loads(f.read())

# Convert JSON to DataFrame
data = pd.DataFrame(
    [
        {'text': row['text'], 'label': row['mappings'][0]['attack_id']}
        for row in data_json['sentences']
        if len(row['mappings']) > 0
    ]
)

# Split data into train and test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter.split(data, data['label']):
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]

# Print the number of rows in each set
print(f"Training set has {len(train_set)} rows and {train_set['label'].nunique()} unique labels.")
print(f"Test set has {len(test_set)} rows and {test_set['label'].nunique()} unique labels.")

# Save the train and test sets into separate files
train_set.to_csv('../data/train_set.csv', index=False)
test_set.to_csv('../data/test_set.csv', index=False)

# Define SciBERT as the learner for our AL framework
model = transformers.BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=data['label'].nunique(),
    output_attentions=False,
    output_hidden_states=False,
)
tokenizer = transformers.BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)
model.train()

# Preprocessing
encoder = OHE(sparse_output=False)
encoder.fit(data[['label']])

def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids


# Initial labelled dataset comprised of 1% of training data
x_labeled, x_unlabeled = train_test_split(train_set, test_size=.99, stratify=train_set['label'])

x_train_labeled = tokenize(x_labeled['text'].tolist())
x_train_unlabeled = tokenize(x_unlabeled['text'].tolist())

y_train_labeled = torch.Tensor(encoder.transform(x_labeled[['label']]))
y_train_unlabeled  = torch.Tensor(encoder.transform(x_unlabeled [['label']]))

# Save preprocessed data and model for use in active learning loop
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'encoder': encoder,
    'x_train_labeled': x_train_labeled,
    'x_train_unlabeled': x_train_unlabeled,
    'y_train_labeled': y_train_labeled,
    'y_train_unlabeled': y_train_unlabeled
}, '../models/preprocessed_data.pth')
