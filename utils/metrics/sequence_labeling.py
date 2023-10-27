"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import, division, print_function
import warnings
from collections import defaultdict
from typing import List, Optional

import numpy as np

from .v1 import  _precision_recall_fscore_support
from .reporters import DictReporter, StringReporter


def precision_recall_fscore_support(y_true: List[List[tuple]],
                                    y_pred: List[List[tuple]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False) :

    def extract_tp_actual_correct(y_true, y_pred, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for i,y_tr in enumerate(y_true):
            for type_name, start, end in y_tr:
                entities_true[type_name].add((i,start, end))
        for i,y_pre in enumerate(y_pred):
            for type_name, start, end in y_pre:
                entities_pred[type_name].add((i,start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def f1_score(y_true: List[List[tuple]], y_pred: List[List[tuple]],
             *,
             average: Optional[str] = 'micro',
             suffix: bool = False,
             sample_weight: Optional[List[int]] = None,
             zero_division: str = 'warn',):

        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     beta=1,
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
        return f


def precision_score(y_true: List[List[tuple]], y_pred: List[List[tuple]],
                    *,
                    average: Optional[str] = 'micro',
                    suffix: bool = False,
                    mode: Optional[str] = None,
                    sample_weight: Optional[List[int]] = None,
                    zero_division: str = 'warn'):

        p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('precision',),
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
        return p


def recall_score(y_true: List[List[tuple]], y_pred: List[List[tuple]],
                 *,
                 average: Optional[str] = 'micro',
                 suffix: bool = False,
                 mode: Optional[str] = None,
                 sample_weight: Optional[List[int]] = None,
                 zero_division: str = 'warn'):

        _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('recall',),
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
        return r


def performance_measure(y_true, y_pred):

    performance_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performance_dict['TP'] = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)
                                 if ((y_t != 'O') or (y_p != 'O')))
    performance_dict['FP'] = sum(((y_t != y_p) and (y_p != 'O')) for y_t, y_p in zip(y_true, y_pred))
    performance_dict['FN'] = sum(((y_t != 'O') and (y_p == 'O'))
                                 for y_t, y_p in zip(y_true, y_pred))
    performance_dict['TN'] = sum((y_t == y_p == 'O')
                                 for y_t, y_p in zip(y_true, y_pred))

    return performance_dict


def classification_report(y_true, y_pred,
                          digits=2,
                          suffix=False,
                          output_dict=False,
                          sample_weight=None,
                          zero_division='warn'):
    target_names_true = {type_name for y_tr in y_true for type_name, _, _ in y_tr}
    target_names_pred = {type_name for y_pre in y_true if y_pre != [] for  type_name, _, _ in y_pre}
    target_names = sorted(target_names_true | target_names_pred)
    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, target_names))
        avg_width = len('weighted avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    # compute per-class scores.
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
        suffix=suffix
    )
    for row in zip(target_names, p, r, f1, s):
        reporter.write(*row)
    reporter.write_blank()

    # compute average scores.
    average_options = ('micro', 'macro', 'weighted')
    for average in average_options:
        avg_p, avg_r, avg_f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
            suffix=suffix
        )
        reporter.write('{} avg'.format(average), avg_p, avg_r, avg_f1, support)
    reporter.write_blank()

    return reporter.report()


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start