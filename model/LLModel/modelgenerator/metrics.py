from typing import Union
import torch
from torchmetrics import Metric
import torchmetrics as tm
from sklearn.metrics import roc_auc_score, average_precision_score


class TopLAcc(Metric):
    def __init__(self, k, **kwargs):
        """Top L accuracy metric for contact prediction

        Args:
            k: int, used to compute top L/k accuracy
        """
        super().__init__(**kwargs)
        self.k = k
        self.add_state(f"acc", default=[], dist_reduce_fx=None)

    def update(self, logits, labels, indices, L):
        _, _, acc = compute_top_l_acc(logits, labels, indices, L // self.k, L)
        self.acc.append(torch.tensor(acc, device=self.device))

    def compute(self):
        return torch.tensor(self.acc, device=self.device).mean()


def compute_top_l_acc(prediction, label, inds, ls, lens):
    """Compute metric for contact prediction for a single sample

    Args:
        prediction: predicted contact probability, tensor of shape (seq_len*seq_len, )
        label: tensor of shape (seq_len*seq_len, )
        inds: tensore of shape (seq_len*seq_len, ), the sorted inds with predicted contact probability from high to low
        ls: for metric, choices are L, L/2, L/5, L/10
        lens: seq length L
    """
    tests = []
    for idx in inds:
        row = idx // lens
        col = idx % lens
        if row >= col:
            continue
        if abs(row - col) <= 6:
            continue
        p = prediction[idx]
        gt = label[idx]
        tests.append((p, gt))
        if len(tests) >= ls:
            break
    cnt = 0
    for p, gt in tests:
        if gt == 1:
            cnt += 1
    return cnt, ls, cnt / ls


class AUROC(Metric):
    def __init__(self):
        """AUROC metric over entire dataset
        Returns:
            tensor: AUROC score over entire dataset
        """
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=0).cpu()
        targets = torch.cat(self.targets, dim=0).cpu()
        return torch.tensor(roc_auc_score(targets, preds))

    def reset(self):
        self.preds = []
        self.targets = []


class AUPRC(Metric):
    def __init__(self):
        """AUPRC metric for entire dataset
        Returns:
            tensor: AUPRC score over entire dataset
        """
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=0).cpu()
        targets = torch.cat(self.targets, dim=0).cpu()
        return torch.tensor(average_precision_score(targets, preds))

    def reset(self):
        self.preds = []
        self.targets = []
