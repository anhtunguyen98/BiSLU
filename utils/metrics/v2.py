import torch
import numpy as np
from .sequence_labeling import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score


def compute_metrics(args, intent_preds, intent_labels, slot_preds, slot_labels,
                    masks, slot_label_set=None):
    slot_labels, slot_preds = get_slot_label(
        slot_labels, slot_preds, masks, slot_label_set)
    slot_precision, slot_recall, slot_f1_score = get_slot_metrics(
        slot_labels, slot_preds)

    intent_preds = intent_preds.detach().cpu().numpy()
    intent_labels = intent_labels.detach().cpu().numpy()
    

    intent_preds = np.array(intent_preds) >=0.5
    intent_acc = get_intent_acc(intent_preds, intent_labels)
        
        
    mean_intent_slot = (intent_acc + slot_f1_score) / 2
    semantic_acc = get_sentence_frame_acc(
        intent_preds, intent_labels, slot_preds, slot_labels)
    result = {
        "intent_acc": intent_acc,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1_score,
        "mean_intent_slot": mean_intent_slot,
        'senmantic_acc': semantic_acc
    }

    return result



def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = intent_preds == intent_labels
    if intent_result.ndim > 1:
        intent_result = np.all(intent_result,axis=1)
    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        one_sent_result = True
        for lb in preds:
            if lb not in labels:
                one_sent_result = False
                break
        if one_sent_result:
            for lb in labels:
                if lb not in preds:
                    one_sent_result = False
                    break
        slot_result.append(one_sent_result)

    slot_result = np.array(slot_result)
    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return semantic_acc


def get_intent_acc(preds, labels):
    return accuracy_score(labels, preds)


def get_slot_metrics(y_true, y_pred):
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), \
        f1_score(y_true, y_pred)


def get_slot_label(labels, preds, masks, label_set):
    y_true = []
    y_pred = []
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        mask = masks[i]
        true_len = int(mask.sum().item())
        pred = pred[:true_len, :true_len]
        label = label[:true_len, :true_len]
        predict_entity, label_entity = get_entities(pred, label, label_set)
        y_true.append(label_entity)
        y_pred.append(predict_entity)
    return y_true, y_pred


def get_entities(input_tensor, label, label_set):
    input_tensor, cate_pred = input_tensor.max(dim=-1)
    predict_entity = get_pred_entity(cate_pred, input_tensor, label_set, True)
    label_entity = get_entity(label, label_set)
    return predict_entity, label_entity


def get_pred_entity(cate_pred, span_scores, label_set, is_flat_ner=True):
    top_span = []
    for i in range(len(cate_pred)):
        for j in range(i, len(cate_pred)):
            if cate_pred[i][j] > 0:
                tmp = (label_set[cate_pred[i][j].item()],
                       i, j, span_scores[i][j].item())
                top_span.append(tmp)
    top_span = sorted(top_span, reverse=True, key=lambda x: x[3])
    res_entity = []
    for t, ns, ne, _ in top_span:
        for _, ts, te, in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are not allowed
                break
        else:
            res_entity.append((t, ns, ne))
    return set(res_entity)


def get_entity(input_tensor, label_set):
    entity = []
    for i in range(len(input_tensor)):
        for j in range(i, len(input_tensor)):
            if input_tensor[i][j] > 0:
                tmp = (label_set[input_tensor[i][j].item()], i, j)
                entity.append(tmp)
    return entity
