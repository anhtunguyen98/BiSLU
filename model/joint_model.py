import torch
import torch.nn as nn
from model.layer import *
from utils.utils import get_soft_slot
from transformers import AutoConfig


class JointModel(nn.Module):
    def __init__(self, args, num_intent_labels, num_slot_labels):
        super(JointModel, self).__init__()
        self.args = args
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.wordrep = WordRep(args)

        self.soft_intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate)

        self.slot_classifier = SlotClassifier(
            config,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.use_intent_context_attention,
            self.args.max_seq_length,
            self.args.dropout_rate,
            self.args.hidden_dim_ffw
        )
        if args.use_soft_slot:
            self.softmax = nn.Softmax(dim=-1)
            hard_intent_input_dim = config.hidden_size + num_slot_labels
            self.hard_intent_classifier = IntentClassifier(
                hard_intent_input_dim, self.num_intent_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, words_lengths, word_attention_mask):

        cls_output, context_embedding = self.wordrep(
            input_ids, attention_mask, words_lengths)

        soft_intent_logits = self.soft_intent_classifier(cls_output)
        biaffine_score, segment_embedding = self.slot_classifier(
            context_embedding, soft_intent_logits, word_attention_mask)

        if self.args.use_soft_slot:
            slot_label_feature = get_soft_slot(biaffine_score, word_attention_mask)
            slot_label_feature = self.softmax(slot_label_feature)

            intent_feature_concat = torch.cat(
                [cls_output, slot_label_feature], dim=-1)
            hard_intent_logits = self.hard_intent_classifier(
                intent_feature_concat)
            return cls_output, segment_embedding, soft_intent_logits, hard_intent_logits, biaffine_score
        else:
            return cls_output, segment_embedding, soft_intent_logits, biaffine_score
