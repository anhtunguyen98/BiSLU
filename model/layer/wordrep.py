import torch
from torch import nn
from transformers import AutoModel

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WordRep(nn.Module):
    def __init__(self, args):
        super(WordRep, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.pooling = MeanPooling()


    def forward(self, input_ids, attention_mask, words_lengths):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=True)

        context_embedding = outputs[0]  # sequence output [b x n x d]

        # Compute align word sub_word matrix
        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_lengths.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx +
                                   sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(
            align_matrix, context_embedding)  # [b x max_length x d]

        cls_output = self.pooling(context_embedding,attention_mask)
        return cls_output, context_embedding_align
