# Install PyTorch first: pip install torch

import torch
import torch.nn as nn

# A tiny training text
words = "the cat sat on the mat the cat ate fish the dog sat on the mat".split()

# Give each word a number
vocabulary = sorted(set(words))
word_to_number = {word: i for i, word in enumerate(vocabulary)}
number_to_word = {i: word for word, i in word_to_number.items()}

# Create examples: use two words to predict the next word
inputs = []
targets = []

for i in range(len(words) - 2):
    inputs.append([word_to_number[words[i]], word_to_number[words[i + 1]]])
    targets.append(word_to_number[words[i + 2]])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)


class NextWordModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.attention = nn.MultiheadAttention(
            embed_dim=16, num_heads=2, batch_first=True
        )
        self.output = nn.Linear(16, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x)

        # Prevent the model from looking at future words
        mask = torch.triu(torch.ones(2, 2), diagonal=1).bool()
        x, _ = self.attention(x, x, x, attn_mask=mask)

        # Use the last word's representation to predict the next word
        return self.output(x[:, -1, :])


model = NextWordModel(len(vocabulary))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
loss_function = nn.CrossEntropyLoss()

# Train the model
for epoch in range(300):
    predictions = model(inputs)
    loss = loss_function(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate: accuracy, top-3 accuracy, and perplexity
with torch.no_grad():
    predictions = model(inputs)
    predicted_words = predictions.argmax(dim=1)

    accuracy = (predicted_words == targets).float().mean().item()
    top3 = predictions.topk(3, dim=1).indices
    top3_accuracy = (top3 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    perplexity = torch.exp(loss_function(predictions, targets)).item()

print(f"Accuracy:     {accuracy:.2%}")
print(f"Top-3 accuracy: {top3_accuracy:.2%}")
print(f"Perplexity:   {perplexity:.2f}")

# Try a next-word prediction
prompt = torch.tensor([[word_to_number["the"], word_to_number["cat"]]])
with torch.no_grad():
    next_word_id = model(prompt).argmax(dim=1).item()

print("Next word after 'the cat':", number_to_word[next_word_id])