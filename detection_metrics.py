from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
from torchvision.ops.boxes import box_iou


def align_coordinates(boxes):
    """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
    Arguments:
        boxes (Tensor[N,4])

    Returns:
        boxes (Tensor[N,4]): aligned box coordinates
    """
    x1y1 = torch.min(boxes[:, :2, ], boxes[:, 2:])
    x2y2 = torch.max(boxes[:, :2, ], boxes[:, 2:])
    boxes = torch.cat([x1y1, x2y2], dim=1)
    return boxes


def calculate_iou_box(gt, pr):
    gt = align_coordinates(gt)
    pr = align_coordinates(pr)

    return box_iou(gt, pr)


def calculate_iou_mask(true_masks: list, pred_masks: torch.tensor):
    IoU_mat = np.zeros((len(true_masks), len(pred_masks)))
    for i in range(len(true_masks)):
        for j in range(pred_masks.shape[0]):
            infer = true_masks[i] * pred_masks[j]
            union = true_masks[i] + pred_masks[j]
            union[union > 0] = 1
            IoU_mat[i, j] = torch.sum(infer) / (torch.sum(union))

    return IoU_mat


def get_mappings(iou_mat):
    mappings = torch.zeros_like(iou_mat)
    gt_count, pr_count = iou_mat.shape

    if not iou_mat[:, 0].eq(0.).all():
        mappings[iou_mat[:, 0].argsort()[-1], 0] = 1

    for pr_idx in range(1, pr_count):
        not_assigned = torch.logical_not(mappings[:, :pr_idx].sum(1))

        targets = not_assigned * iou_mat[:, pr_idx]

        if targets.eq(0).all():
            continue

        pivot = targets.argsort()[-1]
        mappings[pivot, pr_idx] = 1
    return mappings


def tp_calculate(iou_matrix: torch.tensor, labels_gt: torch.tensor, labels_pr: torch.tensor):
    all_labels = torch.unique(torch.concat([labels_gt, labels_pr], dim=0))
    tp = list()
    for label in all_labels:
        gt_index_label = (labels_gt == label).nonzero().squeeze().long()
        pr_index_label = (labels_pr == label).nonzero().squeeze().to(torch.int32).long()
        tp_iou_matrix = iou_matrix[gt_index_label]
        tp_iou_matrix = tp_iou_matrix[:, pr_index_label]
        tp_label = torch.sum(tp_iou_matrix, dim=1)
        tp_label[tp_label > 0] = 1
        tp.append(tp_label.sum() / tp_label.shape[0])
    return np.mean(tp)


def __build_table(iou_mat: torch.tensor, scores: torch.tensor, thresh: float = 0.5) -> pd.DataFrame:
    cumulative_tp = 0
    cumulative_fp = 0
    df_pr = pd.DataFrame(data=[],
                         columns=['conf', 'cumulative_tp', 'cumulative_fp', 'precision', 'recall', 'inter_precision'])
    values, indexes = iou_mat.max(dim=0)
    mat_pred = torch.zeros_like(iou_mat)
    mat_pred[indexes, range(mat_pred.shape[1])] = values
    values, indexes = mat_pred.max(dim=1)
    mat = torch.zeros_like(mat_pred)
    mat[range(mat.shape[0]), indexes] = values
    mat[mat < thresh] = 0
    mat[mat != 0] = 1
    total_gt = mat.shape[0]
    for i in range(mat.shape[1]):
        if mat[:, i].sum() != 0:
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / total_gt
        new_iter = pd.Series(data=[scores[i].tolist(), cumulative_tp, cumulative_fp, precision, recall],
                             index=['conf', 'cumulative_tp', 'cumulative_fp', 'precision', 'recall'], name=i)
        df_pr.loc[i] = new_iter

    return df_pr


def __calculate_map(df_pr: pd.DataFrame):
    inter_recalls = np.arange(.0, 1.1, .1)
    inter_precisions = list()
    for inter_recall in inter_recalls:
        part_df = df_pr.loc[df_pr.recall > inter_recall]
        max_precision = 0
        if part_df.shape[0] != 0:
            max_precision = part_df['precision'].values.max()
        inter_precisions.append(max_precision)

    return np.mean(inter_precisions)


def __calculate_precision(df_pr: pd.DataFrame):
    return df_pr.loc[df_pr.shape[0] - 1, 'precision'].mean()


def __calculate_recall(df_pr: pd.DataFrame):
    return df_pr.loc[df_pr.shape[0] - 1, 'recall'].mean()


def __calculate_IoU(iou_matrix: torch.tensor):
    values, _ = iou_matrix.max(dim=0)
    return torch.mean(values)


def calculate_metric_one_class_box(gt_boxes: torch.tensor,
                                   pr_boxes: torch.tensor,
                                   scores: torch.tensor,
                                   thresh: float = 0.5):
    pr_boxes = pr_boxes[scores.argsort().flip(-1)]
    iou_mat = calculate_iou_box(gt_boxes, pr_boxes)
    IoU = __calculate_IoU(iou_mat)
    df_pr = __build_table(iou_mat, scores, thresh)
    precision = __calculate_precision(df_pr)
    recall = __calculate_recall(df_pr)
    mAP = __calculate_map(df_pr)

    return mAP, precision, recall, IoU


def calculate_metric_one_class_mask(gt_masks: list,
                                    pr_masks: list,
                                    scores: torch.tensor,
                                    thresh: float = 0.5):
    pr_masks = pr_masks[scores.argsort().flip(-1)]
    iou_mat = calculate_iou_mask(gt_masks, pr_masks)
    IoU = __calculate_IoU(iou_mat)
    df_pr = __build_table(iou_mat, scores, thresh)
    precision = __calculate_precision(df_pr)
    recall = __calculate_recall(df_pr)
    mAP = __calculate_map(df_pr)

    return mAP, precision, recall, IoU


class MetricCounter:

    def __init__(self):
        self.__current_map = 0

        self.__hist: dict = {
            'map': list(),
            'precision': list(),
            'recall': list(),
            'IoU': list(),
        }

    def update(self, gt: torch.tensor, labels_gt: torch.tensor,
               pr: torch.tensor, labels_pr: torch.tensor,
               scores: torch.tensor, type_op: str = 'box', thresh=0.5):
        """
        :param gt: torch.tensor - true boxes. Format: [N, 4]
        :param labels_gt: torch.tenser - true labels. Format: [N]
        :param pr: torch.tensor - predicted boxes. Format: [M, 4]
        :param labels_pr: torch.tenser - true labels. Format: [M]
        :param scores: torch.tenser - predicted confidence for labels. Format: [M]
        :param type_op: str - data type: {box, mask}
        :param thresh: float - IoU threshold
        :return: Mean Average Precision [Type: float],
                 Mean Precision [Type: float],
                 Mean Recall [Type: float],
                 Mean IoU [Type: float]
        """

        all_labels = torch.unique(torch.concat([labels_gt, labels_pr], dim=0))

        func_metric = {
            'box': calculate_metric_one_class_box,
            'mask': calculate_metric_one_class_mask
        }

        mAP = list()
        mean_precision = list()
        mean_recall = list()
        mean_IoU = list()

        func = func_metric[type_op]
        for label in all_labels:
            scores_label = scores[labels_pr == label]
            ap, precision, recall, IoU = func(gt, pr, scores_label, thresh)
            mAP.append(ap)
            mean_precision.append(precision)
            mean_recall.append(recall)
            mean_IoU.append(IoU.cpu())

        np.mean(mAP), np.mean(mean_precision), np.mean(mean_recall), np.mean(mean_IoU)
        self.__hist['map'].append(np.mean(mAP))
        self.__hist['precision'].append(np.mean(mean_precision))
        self.__hist['recall'].append(np.mean(mean_recall))
        self.__hist['IoU'].append(np.mean(mean_IoU))

    def get_hist(self, metric_name: str = 'all') -> List[float]:
        """

        :param metric_name: str. Available: map, precision, recall, IoU, all
        :return: List[float]

        """

        if metric_name not in list(self.__hist.keys()) and metric_name != 'all':
            raise ValueError(f'This metric is not supported: {metric_name}')

        return self.__hist if metric_name == 'all' else self.__hist[metric_name]

    def get_metric(self, metric_name: str = 'map') -> Union[Dict[str, float], float]:
        """

        :param metric_name: str. Available: map, precision, recall, IoU, all
        :return: List[float]

        """

        if metric_name not in list(self.__hist.keys()) and metric_name != 'all':
            raise ValueError(f'This metric is not supported: {metric_name}')

        return {key: np.mean(value) for key, value in self.__hist.items()} \
            if metric_name == 'all' \
            else np.mean(self.__hist[metric_name])
