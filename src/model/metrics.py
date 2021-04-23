import torch
from sklearn import metrics


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def overall_accuracy(output, target):
    return metrics.accuracy_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten())


def balanced_accuracy(output, target):
    return metrics.balanced_accuracy_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten())


def kappa(output, target):
    return metrics.cohen_kappa_score(output.data.cpu().numpy().argmax(1).flatten(), target.data.cpu().numpy().flatten(), labels=[0, 1, 2, 3, 4])


def precision(output, target):
    return metrics.precision_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='macro')


def balanced_precision(output, target):
    return metrics.precision_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='weighted')


def overall_precision(output, target):
    return metrics.precision_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='micro')


def recall(output, target):
    return metrics.recall_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='macro')


def balanced_recall(output, target):
    return metrics.recall_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='weighted')


def overall_recall(output, target):
    return metrics.recall_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='micro')


def f1(output, target):
    return metrics.f1_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='macro')


def balanced_f1(output, target):
    return metrics.f1_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='weighted')


def overall_f1(output, target):
    return metrics.f1_score(target.data.cpu().numpy().flatten(), output.data.cpu().numpy().argmax(1).flatten(), labels=[0, 1, 2, 3, 4], average='weighted')
