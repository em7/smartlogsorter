import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans



# Load log file
with open('./logs/err-0.log', 'r') as file:
    lines = file.readlines()

# Example labeling (You should have a labeled dataset)
labels = ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
data = pd.DataFrame({
    'text': lines,
    'label': [labels[i % 3] for i in range(len(lines))]  # Dummy labels
})

# Tokenization and Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y_encoded, range(len(y_encoded)), test_size=0.2, random_state=42
)


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.indices = indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


train_dataset = LogDataset(X_train, y_train, train_indices)
test_dataset = LogDataset(X_test, y_test, test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


class LogClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(labels)

model = LogClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch, _ in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f'Accuracy: {100 * correct / total}%')


def extract_features_with_indices(model, loader):
    features = []
    indices = []
    model.eval()
    with torch.no_grad():
        for X_batch, _, idx in loader:
            output = model.fc1(X_batch)
            features.append(output.numpy())
            indices.extend(idx.numpy())
    return np.concatenate(features, axis=0), indices

features, feature_indices = extract_features_with_indices(model, train_loader)


# Apply K-Means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(features)

# Map clusters to original data
train_dataset_clustered = pd.DataFrame({'text': data['text'].iloc[feature_indices], 'cluster': clusters})
print(train_dataset_clustered)
