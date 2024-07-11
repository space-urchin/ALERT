import torch
import transformers
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from utils import apply_attention_mask

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model, encoder, and tokenizer with map_location
setup = torch.load("models/preprocessed_data.pth", map_location=device)
encoder = setup['encoder']
tokenizer = setup['tokenizer']

# Initialize the model
model = transformers.BertForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=encoder.categories_[0].size,
    output_attentions=False,
    output_hidden_states=False,
)

# Load the second checkpoint with map_location
checkpoint = torch.load("models/alert_coreset_200_cycles.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)

# Load test data
test_set = pd.read_csv('data/test_set.csv')

# Preprocess test data
def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

x_test = tokenize(test_set['text'].tolist())

model.to(device)
model.eval()

preds = []
batch_size = 20

# Perform predictions on the test set
with torch.no_grad():
    for i in range(0, x_test.shape[0], batch_size):
        x = x_test[i : i + batch_size].to(device)
        out = model(x, attention_mask=apply_attention_mask(x, tokenizer))
        preds.extend(out.logits.to('cpu'))

predicted_labels = (
    encoder.inverse_transform(
        F.one_hot(
            torch.vstack(preds).softmax(-1).argmax(-1),
            num_classes=encoder.categories_[0].size
        )
        .numpy()
    )
    .reshape(-1)
)

# Calculate evaluation metrics
predicted = list(predicted_labels)
actual = test_set['label'].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
