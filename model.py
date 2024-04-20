import collections
import pprint
import re
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import track
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as confusion_matrix_f
from sophia.sophia import SophiaG
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM Linear输入*2

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Model:
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_classes: int):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model: BiLSTM | None = None
        self.dataset: TensorDataset | None = None

    def create_model(self):
        self.model = BiLSTM(self.vocab_size, self.embedding_size, self.hidden_size, self.num_classes).to(device=DEVICE)

    def train(self, epochs: int, batch_size: int, optimizer: Literal['Adagrad', 'Adam', 'Sophia', 'SDG']):
        if self.dataset is None:
            raise ValueError("Dataset is None")
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer_dic = {
            'SDG': optim.SGD(self.model.parameters()),
            'Adagrad': optim.Adagrad(self.model.parameters()),
            'Adam': optim.Adam(self.model.parameters()),
            'Sophia': SophiaG(self.model.parameters())
        }

        optimizer = optimizer_dic[optimizer]

        for epoch in range(epochs):
            for inputs, targets in track(DataLoader(dataset=self.dataset,
                                                    batch_size=batch_size),
                                         description="Epoch {}/{}".format(epoch + 1, epochs)):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.view(-1, 5), targets.float().view(-1, 5))
                loss.backward()
                optimizer.step()
            true_label = []
            predicted_label = []
            for i in range(outputs.shape[0]):
                for j in range(outputs.shape[1]):
                    if np.argmax(targets[i, j].cpu().detach().numpy()) != 4:
                        true_label.append(np.argmax(targets[i, j].cpu().detach().numpy()))
                        predicted_label.append(np.argmax(outputs[i, j].cpu().detach().numpy()))
            print(f'Loss: {loss.item()}, accuracy: {accuracy_score(predicted_label, true_label)}')

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)

    def show(self):
        summary(self.model, (32,), dtypes=[torch.long])


class Data:
    def __init__(self, max_len):
        self.max_len = max_len
        self.vocab_size = 0
        self.char2id = None
        self.tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}
        self.id2tags = {0: 's', 1: 'b', 2: 'm', 3: 'e', 4: 'x'}

    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            vocab = f.read().rstrip('\n').split('\n')
            vocab = list(''.join(vocab))
            stat = {}
            for v in vocab:
                stat[v] = stat.get(v, 0) + 1
            stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
            vocab = [s[0] for s in stat]
            self.char2id = {w: c + 1 for c, w in enumerate(vocab)}
            self.vocab_size = len(vocab)

    def load_data(self, path: str) -> torch.tensor:
        data = open(path, encoding='utf-8').read().rstrip('\n')
        data = re.split('[，。？！、\n]', data)
        transition_counts = collections.defaultdict(int)
        transition_source = collections.defaultdict(int)
        X_data = []
        Y_data = []
        for sentence in data:
            sentence = sentence.split(" ")
            X = []
            y = []
            try:
                for s in sentence:
                    s = s.strip()
                    if len(s) == 0:
                        continue
                    elif len(s) == 1:
                        X.append(self.char2id[s])
                        y.append(self.tags['s'])
                    elif len(s) > 1:
                        X.append(self.char2id[s[0]])
                        y.append(self.tags['b'])
                        for i in range(1, len(s) - 1):
                            X.append(self.char2id[s[i]])
                            y.append(self.tags['m'])
                        X.append(self.char2id[s[-1]])
                        y.append(self.tags['e'])
                for i in range(len(y) - 1):
                    transition = self.id2tags[y[i]]+self.id2tags[y[i + 1]]
                    transition_source[self.id2tags[y[i]]] += 1
                    transition_counts[transition] += 1
                if len(X) > self.max_len:
                    X = X[:self.max_len]
                    y = y[:self.max_len]
                else:
                    for i in range(self.max_len - len(X)):
                        X.append(0)
                        y.append(self.tags['x'])
            except:
                continue
            else:
                if len(X) > 0:
                    X_data.append(X)
                    Y_data.append(y)
        X_data = torch.tensor(X_data).to(device=DEVICE)

        Y_data = F.one_hot(torch.tensor(Y_data), num_classes=5).to(device=DEVICE)

        transition_probabilities = {}
        for transition, count in transition_counts.items():
            transition_probabilities[transition] = count / transition_source[transition[0]]

        return X_data, Y_data, transition_probabilities


class Metrics:
    def __init__(self, y_true: torch.tensor, y_pred: torch.tensor, num_classes):
        true_label = []
        predicted_label = []
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if np.argmax(y_true[i, j].cpu().detach().numpy()) != 4:
                    true_label.append(np.argmax(y_true[i, j].cpu().detach().numpy()))
                    predicted_label.append(np.argmax(y_pred[i, j].cpu().detach().numpy()))
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.num_classes = num_classes

    def show(self, path: str | None = None):
        confusion_matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        for true, pred in zip(self.true_label, self.predicted_label):
            confusion_matrix[true][pred] += 1

        precision = []
        recall = []
        f1_score = []
        support = []
        for i in range(self.num_classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(self.num_classes) if j != i)
            fn = sum(confusion_matrix[i][j] for j in range(self.num_classes) if j != i)

            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            f1_score.append(
                2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0)
            support.append(tp)
        format_string = "{:." + str(2) + "f}"
        correct = sum(confusion_matrix[i][i] for i in range(self.num_classes))
        total = sum(sum(confusion_matrix[i]) for i in range(self.num_classes))
        accuracy = correct / total
        print("Confusion Matrix :")
        pprint.pprint(confusion_matrix)
        report = {
            "precision": [float(format_string.format(p)) for p in precision],
            "recall": [float(format_string.format(r)) for r in recall],
            "f1_score": [float(format_string.format(f)) for f in f1_score],
            "support": support,
        }
        report = pd.DataFrame(report)
        print(report)
        print("Accuracy Score :", accuracy)
        if path is not None:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Confusion Matrix :\n")
                f.write(pprint.pformat(confusion_matrix) + '\n')
                f.write(str(report) + '\n')
                f.write(f"Accuracy Score :{accuracy}")

    def use_sklearn(self):
        print("Confusion Matrix :")
        print(confusion_matrix_f(self.true_label, self.predicted_label))
        print("Classification Report :")
        print(classification_report(self.true_label, self.predicted_label, digits=2))
        print("Accuracy ", accuracy_score(self.true_label, self.predicted_label))
