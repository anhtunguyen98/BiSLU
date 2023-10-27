import torch
from torch.utils.data import Dataset
import os
from utils.metrics.sequence_labeling import get_entities
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputSample(object):

    def __init__(self, folder, max_length):
        self.folder = folder
        self.input_text_file = "seq.in"
        self.intent_label_file = "label"
        self.slot_labels_file = "seq.out"
        self.max_length = max_length

    def read_file(self, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:

                lines.append(line.strip())
            return lines

    def to_json_string(self,):
        """Convert samples and fix max length"""

        texts = self.read_file(os.path.join(self.folder, self.input_text_file))
        intent_labels = self.read_file(
            os.path.join(self.folder, self.intent_label_file))
        slot_labels = self.read_file(os.path.join(
            self.folder, self.slot_labels_file))

        samples = []
        for text, intent_label, slot_label in zip(texts, intent_labels, slot_labels):

            text = text.split()
            slot_label = slot_label.split()
            if len(slot_label) > self.max_length:
                slot_label = slot_label[:self.max_length]
                text = text[:self.max_length]
            sample = {'text': text,
                      'intent_label': intent_label,
                      'slot_label': get_entities(slot_label)}
            samples.append(sample)
        return samples


class MyDataSet(Dataset):

    def __init__(self, args, folder,
                 intent_label_set, slot_label_set, tokenizer):

        self.args = args
        self.samples = InputSample(
            folder=folder, max_length=args.max_seq_length).to_json_string()
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length + 2

        self.slot_label_id = {w: i for i, w in enumerate(slot_label_set)}
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}

    def preprocess(self, tokenizer, sentence):
        """Preprocess a input sentence"""

        tokens, words_lengths = [], []
        for i, word in enumerate(sentence):

            # tokenize a word and process input
            token = tokenizer.tokenize(word)

            if not token:
                token = [tokenizer.unk_token]
            tokens.extend(token)
            words_lengths.append(len(token))

        tokens += [tokenizer.sep_token]
        words_lengths += [1]

        tokens = [tokenizer.cls_token] + tokens
        words_lengths = [1] + words_lengths

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        word_attention_mask = [1] * len(words_lengths)

        padding_length = self.max_seq_length - len(word_attention_mask)

        if padding_length > 0:
            word_attention_mask += [0] * padding_length
            words_lengths = words_lengths + ([1]*padding_length)

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(words_lengths), torch.tensor(word_attention_mask)

    def span_maxtrix_label(self, label):
        start, end, entity = [], [], []
        for lb in label:
            start.append(lb[1]+1)
            end.append(lb[2]+1)
            if lb[0] in self.slot_label_id:
                entity.append(self.slot_label_id[lb[0]])
            else:
                entity.append(self.slot_label_id['UNK'])

        label = torch.sparse.FloatTensor(torch.tensor([start, end], dtype=torch.int64), torch.tensor(entity),
                                         torch.Size([self.max_seq_length, self.max_seq_length])).to_dense()
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        sentence = sample['text']
        slot_label = sample['slot_label']
        intent_label = sample['intent_label']

        intent_label_txt = intent_label.split('#')
        intent_label = [0] * len(self.intent_label_id)
        for intent in intent_label_txt:
            if intent in self.intent_label_id:
                intent_label[self.intent_label_id[intent]] = 1
            else:
                intent_label[self.intent_label_id['UNK']] = 1


        input_ids, attention_mask, words_lengths, word_attention_mask = self.preprocess(
            self.tokenizer, sentence)

        slot_label = self.span_maxtrix_label(slot_label)

        return input_ids, attention_mask, words_lengths, word_attention_mask,\
            torch.tensor(intent_label), slot_label.long(
            )

    def __len__(self):
        return len(self.samples)


def pad_concat(sample, pad_value=0):
    final_tensor = []
    max_length = max([i.size(0) for i in sample])
    for i in sample:
        i = i.long()
        pa = max_length - i.size(0)
        if pa > 0:
            tensor = torch.nn.functional.pad(i, (0, pa), "constant", pad_value)
            final_tensor.append(tensor)
        else:
            final_tensor.append(i)
    return torch.stack(final_tensor)


def collate_fn(data, pad_id):

    input_ids, attention_mask, words_lengths, word_attention_mask, intent_label, slot_label = zip(
        *data)

    input_ids = pad_concat(input_ids, pad_id)
    attention_mask = pad_concat(attention_mask, 0)

    return input_ids, attention_mask, torch.stack(words_lengths), torch.stack(word_attention_mask), torch.stack(intent_label), torch.stack(slot_label)
