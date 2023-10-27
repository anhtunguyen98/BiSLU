import numpy as np
import torch
import torch.nn as nn
from model.layer import FeedforwardLayer, BiaffineLayer



class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(
            self,
            config,
            num_intent_labels,
            num_slot_labels,
            use_intent_context_attn=False,
            max_seq_len=50,
            dropout_rate=0.0,
            hidden_dim_ffw=300,
    ):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels

        if self.use_intent_context_attn:
            hidden_dim = config.hidden_size + num_intent_labels
            self.sigmoid = nn.Sigmoid(dim=-1)
        else:
            hidden_dim = config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.feedStart = FeedforwardLayer(
            d_in=hidden_dim, d_hid=hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=hidden_dim, d_hid=hidden_dim_ffw)
        self.biaffine = BiaffineLayer(
            inSize1=hidden_dim, inSize2=hidden_dim, classSize=256)
        self.classifier = nn.Linear(256,num_slot_labels)

    def forward(self, word_context, intent_context):

        if self.use_intent_context_attn:
            intent_context = self.sigmoid(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            context = intent_context.repeat(1,word_context.shape[1],1)
            output = torch.cat((context, word_context), dim=2)
        else:
            output = word_context
        x = self.dropout(output)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        embedding = self.biaffine(start, end)
        embedding = self.dropout(embedding)
        score = self.classifier(embedding)
        return score, embedding
