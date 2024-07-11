import torch
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
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
    output_hidden_states=False,
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
num_annotation_rounds = 50
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

    # Use the trained model to make predictions on the unlabeled set
    confidence_scores = []
    with torch.no_grad():
        model.eval()
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x_batch_unlabeled = x_train_unlabeled[i : i + batch_size]
            out = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled, tokenizer))
            logits = out.logits
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
            confidence_scores.extend(entropy)

    sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)
    selected_indices = sorted_indices[:num_samples_to_select]

    # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices]])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices]])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]

filename = "../models/model_max_entropy_" + str(num_annotation_rounds) + ".pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder': encoder,
    'tokenizer': tokenizer,
}, filename)
