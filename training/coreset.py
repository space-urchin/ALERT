
import torch
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
import transformers
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import sys
sys.path.append("..")
from utils import apply_attention_mask


# Load preprocessed data and model
checkpoint = torch.load('../models/preprocessed_data.pth')
model = transformers.BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=checkpoint['encoder'].categories_[0].size,
    output_attentions=False,
    output_hidden_states=True,
)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = checkpoint['tokenizer']
encoder = checkpoint['encoder']
x_train_labeled = checkpoint['x_train_labeled']
x_train_unlabeled = checkpoint['x_train_unlabeled']
y_train_labeled = checkpoint['y_train_labeled']
y_train_unlabeled = checkpoint['y_train_unlabeled']


# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x_train_labeled = x_train_labeled.to(device)
x_train_unlabeled = x_train_unlabeled.to(device)
y_train_labeled = y_train_labeled.to(device)
y_train_unlabeled = y_train_unlabeled.to(device)

# Active Learning loop
optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 2
num_epochs = 2
batch_size = 10

for annotation_round in range(num_annotation_rounds):
    print(f"Annotation Round {annotation_round + 1}")
    
    # Train the model on the currently labeled set of data
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size), desc=f"Epoch {epoch + 1}"):
            x = x_train_labeled[i : i + batch_size]
            y = y_train_labeled[i : i + batch_size]
            model.zero_grad()
            out = model(x, attention_mask=apply_attention_mask(x, tokenizer), labels=y)
            epoch_losses.append(out.loss.item())
            out.loss.backward()
            optim.step()
        print(f"Epoch {epoch + 1} loss: {mean(epoch_losses)}")

    cls_vector_labelled = []
    with torch.no_grad():
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size)):
            x = x_train_labeled[i : i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector_labelled.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector_labelled = torch.stack(cls_vector_labelled)

    cls_vector_unlabelled = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, x_train_unlabeled.shape[0], batch_size)):
            x = x_train_unlabeled[i : i + batch_size]
            model.zero_grad()
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector_unlabelled.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector_unlabelled = torch.stack(cls_vector_unlabelled)
    
    a = cls_vector_unlabelled
    b = cls_vector_labelled
 
    selected_vectors = []
    selected_indices = []
    for _ in range(10):

        distances = pairwise_distances_argmin_min(a, b)[1]
        max_distance_index = torch.argmax(torch.tensor(distances))
        selected_vectors.append(a[max_distance_index.item()])
        b = torch.cat((b, a[max_distance_index].unsqueeze(0)))
        a = torch.cat((a[:max_distance_index], a[max_distance_index+1:]))

    for vector in selected_vectors:
        for i, cls_vector in enumerate(cls_vector_unlabelled):
            if torch.equal(vector, cls_vector):
                selected_indices.append(i)
                break

    # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices]])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices]])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]
    
filename = "../models/model_coreset" + str(num_annotation_rounds) + ".pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder': encoder,
    'tokenizer': tokenizer,
}, filename)
