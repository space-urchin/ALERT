import torch
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import torch.nn as nn
import torch.nn.functional as F
import transformers
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

# Active Learning loop
optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 50
num_inference_cycles = 10
num_epochs = 10
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

    cls_vector = []
    with torch.no_grad():
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size)):
            x = x_train_labeled[i : i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector = torch.stack(cls_vector)
    print(cls_vector.shape)

    print(f"Training MLP")
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, cls_vector.shape[0], 10):
            p_model.train().to('cpu')
            x = cls_vector[i : i + 10]
            y = y_train_labeled[i : i + 10].to('cpu')
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
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x_batch_unlabeled = x_train_unlabeled[i : i + batch_size]
            model.zero_grad()
            outputs = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector_eval.extend(last_hidden_states[:, 0, :].to('cpu'))

    entropies = []
    cls_vector_eval = torch.stack(cls_vector_eval)

    # Repeat inference 10 times
    for _ in range(num_inference_cycles):
        preds=[]
        with torch.no_grad():
            p_model.eval().to(device)
            p_model.dropout.train()
            for i in range(0, cls_vector_eval.shape[0], 10):
                x_batch_unlabeled = cls_vector_eval[i : i + 10].to(device)
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


filename = "../models/model_mcd_" + str(num_annotation_rounds) + ".pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder': encoder,
    'tokenizer': tokenizer,
}, filename)