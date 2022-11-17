import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true, y_true_weighted):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        wtp = ((y_true * y_true_weighted[:, None]) * y_pred).sum(dim=0).to(torch.float32)
        wfn = ((y_true * y_true_weighted[:, None]) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        
        precision = tp / (tp + fp + self.epsilon)
        weighted_recall = wtp / (wtp + wfn + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        weighted_f1 = (2*precision*weighted_recall) / (precision + weighted_recall + self.epsilon)
        weighted_f1 = weighted_f1.clamp(min=self.epsilon, max=1-self.epsilon)
        
        f1 = (2*precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return f1

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1  = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2( F.relu(self.linear1(x)))
