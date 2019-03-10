from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import configs.basic_config as args
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score


def print_conf_matrix(y_pre, y_true):
    _, y_pred = torch.max(y_pre.data, 1)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    print("confusion_maxtrix, left labels is y_true, up labels is y_pre")
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print('\t' + '\t'.join(list(map(str, labels))))

    for i in range(len(conf_mat)):
        print(str(i) + '\t' + "\t".join(list(map(str, conf_mat[i]))))


def loss_fn(logits, labels):
    loss_f = CrossEntropyLoss()
    loss = loss_f(logits.view(-1, len(args.labels)), labels.view(-1))
    return loss


def evaluate(y_pred, y_true, train=True):
    if train:
        _, y_pred = torch.max(y_pred.data, 1)
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return acc, f1
    else:
        best_f1 = 0.0
        best_acc = 0.0
        best_threshold = 0.5
        y_pred_pos = F.softmax(y_pred, dim=1)
        y_pred_pos = y_pred_pos.data.numpy()[:, 1]
        for i in range(10, 61):
            threshold = i * 0.01
            y_pred_ = np.array([1 if v > threshold else 0 for v in y_pred_pos])
            f1 = f1_score(y_true, y_pred_, average='macro')
            acc = accuracy_score(y_true, y_pred_)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_threshold = threshold

        return best_acc, best_f1, best_threshold


def class_report(y_pred, y_true):
    _, y_pred = torch.max(y_pred.data, 1)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    classify_report = classification_report(y_true, y_pred)
    print('\n\nclassify_report:\n', classify_report)
