import torch
from torch import Tensor
from typing import Tuple


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)
        self.index = 0

    def update_hist(self, hist):
        self.hist += hist.to(self.hist.device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        self.index = self.index + 1
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(
            self.num_classes, self.num_classes
        )

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()] = 0.0
        miou = ious.mean().item()
        # miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()] = 0.0
        mf1 = f1.mean().item()
        # mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()] = 0.0
        macc = acc.mean().item()
        # macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
        
    def compute_precision_recall(self) -> Tuple[Tuple[float, float], Tuple[Tensor, Tensor]]:
        """
        Returns:
            (precision_class1, recall_class1), ([precision_per_class], [recall_per_class])
        """
        TP = self.hist.diag()
        FP = self.hist.sum(0) - TP  # Predicted as class i but not actually class i
        FN = self.hist.sum(1) - TP  # Actually class i but predicted as something else

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        precision[precision.isnan()] = 0.0
        recall[recall.isnan()] = 0.0

        precision *= 100
        recall *= 100
        
        print("precision",precision.cpu().numpy().round(2).tolist())
        print("recall",recall.cpu().numpy().round(2).tolist())

        # Assuming class 1 is the "apple" class
        return (round(precision[1].item(), 2), round(recall[1].item(), 2)), (
            precision.cpu().numpy().round(2).tolist(),
            recall.cpu().numpy().round(2).tolist()
        )
