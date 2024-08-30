import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
from annoy import AnnoyIndex
import numpy as np
import os
from tqdm import tqdm

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=0.6)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x

# Function to get frequent and unique words for each type
def get_frequent_and_unique_words(df, mbti_type, top_n=50):
    type_df = df[df['type'] == mbti_type]
    words = ' '.join(type_df['posts']).split()
    word_counts = Counter(words)
    most_common_words = [word for word, count in word_counts.most_common(top_n)]
    unique_words = [word for word, count in word_counts.items() if count == 1]
    return most_common_words, unique_words

# Function to construct graphs from the new DataFrame
def construct_graph(row):
    G = nx.Graph()
    type_words = row['frequent_words'] + row['unique_words']
    G.add_nodes_from(type_words)
    for i, word1 in enumerate(type_words):
        for j, word2 in enumerate(type_words):
            if i != j:
                G.add_edge(word1, word2, weight=1)
    pyg_graph = from_networkx(G)
    return pyg_graph

def construct_graphs_from_new_words_df(new_df):
    graphs = new_df.apply(construct_graph, axis=1)
    return list(graphs)

# Function to create word embeddings and construct graphs
def process_data_for_type(mbti_type, df, encoded_labels):
    type_df = df[df['type'] == mbti_type]
    sentences = [post.split() for post in type_df['posts']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Collect word vectors and their contextual relationships
    word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    words = list(word_vectors.keys())
    
    # Create Annoy Index for fast similarity search
    f = 100
    t = AnnoyIndex(f, 'angular')
    for i, word in enumerate(words):
        t.add_item(i, word_vectors[word])
    t.build(10)
    
    # Construct graphs with contextual relationships
    G = nx.Graph()
    for word in words:
        G.add_node(word)
        neighbors = t.get_nns_by_item(words.index(word), 10, include_distances=True)
        for neighbor_idx, dist in zip(neighbors[0], neighbors[1]):
            if dist <= 0.5:
                G.add_edge(word, words[neighbor_idx], weight=1 - dist)
    
    pyg_graph = from_networkx(G)
    labels = torch.tensor(encoded_labels[type_df.index], dtype=torch.long)
    return pyg_graph, labels

def process_dataset(fpath):
    df = pd.read_csv(fpath)
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])
    mbti =[]
    freq_w = []
    uniq_w = []
    for mbti_type in df['type'].unique():
        frequent_words, unique_words = get_frequent_and_unique_words(df, mbti_type)
        mbti.append(mbti_type)
        freq_w.append(frequent_words)
        uniq_w.append(unique_words)
    
    new_df = pd.DataFrame({
        'type': mbti,
        'frequent_words': freq_w,
        'unique_words': uniq_w
    })

    print(new_df.head(5))

    # Construct graphs and labels for all types
    graphs = []
    labels_list = []
    for mbti_type in df['type'].unique():
        graph, labels = process_data_for_type(mbti_type, df, label_encoder.transform(df['type']))
        graphs.append(graph)
        labels_list.append(labels)

    # Flatten the lists of graphs and labels
    labels = torch.cat(labels_list)

    dataset = GraphDataset(graphs, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return new_df, dataset, dataloader,graphs,labels

'''def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, labels in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)'''

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False, ncols=100)
    for data, labels in pbar:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in dataloader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(dataloader.dataset)

def train_with_5_fold_cross_validation(dataset, model, optimizer, criterion, save_path):
    best_accuracy = 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

        for epoch in range(50):
            train_loss = train(model, train_loader, optimizer, criterion)
            accuracy = evaluate(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), save_path)
            print(f'Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}')


def visualize_graphs(graphs, types, label_encoder, save_path='/content/drive/MyDrive/dataset/graph_visualizations'):
    for i, graph in enumerate(graphs):
        plt.figure(figsize=(10, 7))
        nx.draw_networkx(nx.Graph(graph), with_labels=True)
        plt.title(f'Type {label_encoder.inverse_transform([types[i]])[0]}')
        plt.savefig(f'{save_path}/type_{label_encoder.inverse_transform([types[i]])[0]}.png')
        plt.close()

def main():
    fpath = r"C:\Users\HP\Downloads\archive (39)\MBTI 500.csv"
    label_encoder = LabelEncoder()
    new_df, dataset, dataloader, graphs, labels = process_dataset(fpath)
    save_path = r"C:\Users\HP\OneDrive\Documents\GitHub\Mbti_type"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATClassifier(in_dim=100, hidden_dim=64, num_heads=4, n_classes=len(new_df['type'].unique())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_with_5_fold_cross_validation(dataset, model, optimizer, criterion, save_path)
    visualize_graphs(graphs, labels, label_encoder, save_path)

if __name__ == "__main__":
    main()








