from utils.metrics import compute_metrics
from utils.dataloader import collate_fn
from utils.early_stopping import EarlyStopping
from utils.trainer_state import TrainerState
from utils.scl_loss import scl_intent_loss_func, scl_slot_loss_func
from utils.mld import MLD
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import JointModel
from tqdm import trange
from utils.utils import get_useful_ones, get_mask, get_useful_embedding
import os
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)



def intent_loss_func(y_hat, y_true, reduction='mean'):
    """
    :param y_hat: model predictions, shape(batch, classes)
    :param y_true: target labels (batch, classes)
    :param reduction: whether to avg or sum loss
    :return: loss
    """
    loss = torch.zeros(y_true.size(0)).cuda()
    for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
        y_z, y_o = (y == 0).nonzero(), y.nonzero()
        if y_o.nelement() != 0:
            output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
            num_comparisons = y_z.size(0) * y_o.size(0)
            loss[idx] = output.div(num_comparisons)
    return loss.mean() if reduction == 'mean' else loss.sum()


class Trainer(object):
    def __init__(self,
                 args,
                 tokenizer=None,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 intent_label_set=None,
                 slot_label_set=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.model = JointModel(args=args,
                                num_intent_labels=len(intent_label_set),
                                num_slot_labels=len(slot_label_set))
        self.model.to(self.device)
        self.train_dataset = train_dataset
        self.w_drop_out = [0.3, 0.2]
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.intent_label_set = intent_label_set
        self.slot_label_set = slot_label_set
        self.trainer_state = TrainerState()

    def get_train_dataloader(self):
        sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=sampler,
                                      num_workers = 32,
                                      batch_size=self.args.train_batch_size,
                                      collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id))
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset):
        sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=sampler,
                                     num_workers = 32,
                                     batch_size=self.args.eval_batch_size,
                                     collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id))
        return eval_dataloader

    def set_dropout_mf(
        self, 
        model, 
        w
        ):
        """Alters the dropouts in the embeddings.
        """
        # ------ set hidden dropout -------#
        if hasattr(model, 'module'):
            model.wordrep.bert.module.embeddings.dropout.p = w
            for i in model.wordrep.bert.module.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w        
        else:
            model.wordrep.bert.embeddings.dropout.p = w
            for i in model.wordrep.bert.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w
            
        return model


    def compute_scl_loss(self, model, cls_output, segment_embedding, slot_labels, intent_labels, mask, inputs):


        if self.args.loss_coef_intent_scl != 0:
            cls_output = cls_output.unsqueeze(1)
        
        if self.args.loss_coef_slot_scl != 0:

            segment_embedding = get_useful_embedding(segment_embedding,mask).unsqueeze(1)

        # ---- iteratively create dropouts -----#
        for p_dpr in self.w_drop_out:
        # -- Set models dropout --#
            model = self.set_dropout_mf(model, w=p_dpr)
        # ---- concat logits ------#
            positive_cls_output, positive_segment_embedding, _, _, _ = model(**inputs)

            if self.args.loss_coef_intent_scl != 0:
                positive_cls_output = positive_cls_output.unsqueeze(1)
                cls_output = torch.cat((cls_output,positive_cls_output), 1)

            if self.args.loss_coef_slot_scl != 0:
                positive_segment_embedding = get_useful_embedding(positive_segment_embedding,mask)
                positive_segment_embedding = positive_segment_embedding.unsqueeze(1)
                segment_embedding = torch.cat((segment_embedding,positive_segment_embedding), 1)


        total_loss = 0

        if self.args.loss_coef_intent_scl != 0:
            intent_scl_loss = scl_intent_loss_func(cls_output,intent_labels)
            total_loss += self.args.loss_coef_intent_scl * intent_scl_loss
        
        if self.args.loss_coef_slot_scl != 0:

            slot_scl_loss = scl_slot_loss_func(segment_embedding,slot_labels)
            total_loss +=  self.args.loss_coef_slot_scl * slot_scl_loss
        return total_loss

    def compute_loss(self, model, inputs, slot_labels, intent_labels, mask):
        cls_output, segment_embedding, soft_intent_logits, final_intent_logits, biaffine_score = model(**inputs)

        intent_loss = intent_loss_func(
            final_intent_logits, intent_labels.float())
        slot_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        total_loss = 0

        if self.args.use_sd:

            # sd_loss_func = torch.nn.MSELoss(reduction='mean')
            sd_loss_func = MLD()
            sd_loss = sd_loss_func(soft_intent_logits, final_intent_logits) 
            total_loss += self.args.sd_loss_coef * sd_loss 

        masks = get_mask(mask=mask)
        masks = masks.to(self.device)
        tmp_out, tmp_label = get_useful_ones(
            biaffine_score, slot_labels, masks)
        slot_loss = slot_loss_func(tmp_out, tmp_label)

        total_loss += self.args.loss_coef_intent * intent_loss + self.args.loss_coef_slot * slot_loss

        if self.args.use_scl:

            total_loss += self.compute_scl_loss(model, cls_output, segment_embedding, tmp_label, intent_labels, masks, inputs)

        return total_loss, final_intent_logits, biaffine_score
    
    def train(self):
        train_dataloader = self.get_train_dataloader()
        total_steps = len(
            train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.args.warmup_proportion*total_steps), num_training_steps=total_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d",
                    self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_steps)

        global_step = 0
        self.model.zero_grad()
        early_stopping = EarlyStopping(
            patience=self.args.early_stopping, verbose=True)
        self.trainer_state.num_train_epochs = self.args.num_train_epochs

        self.model.zero_grad()
        self.model.train()
        for epoch in trange(self.args.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")

            train_loss = 0
            print('\nEPOCH:', epoch)
            self.model.train()
            for step, batch in enumerate(train_dataloader):

                inputs = {'input_ids': batch[0].to(self.device),
                          'attention_mask': batch[1].to(self.device),
                          'words_lengths': batch[2].to(self.device),
                          'word_attention_mask': batch[3].to(self.device),
                          }
                intent_labels = batch[4].to(self.device)
                slot_labels = batch[5].to(self.device)
                self.optimizer.zero_grad()

                total_loss, intent_logit, biaffine_score = self.compute_loss(
                    self.model, inputs, slot_labels, intent_labels, batch[3])

                if self.args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.args.gradient_accumulation_steps

                train_loss += total_loss.item()
                total_loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                    global_step += 1

                    self.trainer_state.epoch = epoch
                    self.trainer_state.global_step = global_step
                    self.trainer_state.max_steps = total_steps
                    self.trainer_state.loss = train_loss/(step+1)
                if (step + 1) % self.args.logging_steps == 0:
                    logger.info('\n%s', self.trainer_state.to_string())
            results = self.evaluate('dev')
            early_stopping(results[self.args.tuning_metric], self.args)
            if early_stopping.counter == 0:
                self.save_model()
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, mode):
        if mode == "test":
            eval_dataset = self.test_dataset
        elif mode == "dev":
            eval_dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()

        eval_loss = 0
        intent_labels, intent_preds, slot_labels, slot_preds, word_masks = [], [], [], [], []
        for step, batch in enumerate(eval_dataloader):

            with torch.no_grad():

                inputs = {'input_ids': batch[0].to(self.device),
                          'attention_mask': batch[1].to(self.device),
                          'words_lengths': batch[2].to(self.device),
                          'word_attention_mask': batch[3].to(self.device),
                          }
                intent_label = batch[4].to(self.device)
                slot_label = batch[5].to(self.device)

                total_loss, intent_logit, biaffine_score = self.compute_loss(
                    self.model, inputs, slot_label, intent_label, batch[3])
            eval_loss += total_loss.item()

            intent_labels.append(intent_label)
            intent_preds.append(intent_logit)
            slot_labels.append(slot_label)
            slot_preds.append(biaffine_score)
            word_masks.append(batch[3])

        # compute metrics
        intent_labels = torch.cat(intent_labels, dim=0)
        intent_preds = torch.cat(intent_preds, dim=0)
        slot_labels = torch.cat(slot_labels, dim=0)
        slot_preds = torch.cat(slot_preds, dim=0)
        word_masks = torch.cat(word_masks, dim=0)

        eval_loss = eval_loss / len(eval_dataloader)
        results = {"loss": eval_loss}
        results.update(compute_metrics(self.args, intent_preds, intent_labels,
                       slot_preds, slot_labels, word_masks, self.slot_label_set))

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)

        return results

    def save_model(self):
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      }
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        path = os.path.join(self.args.output_dir, 'checkpoint.pth')
        torch.save(checkpoint, path)
        torch.save(self.args, os.path.join(
            self.args.output_dir, 'training_args.bin'))
        self.trainer_state.save_to_json(os.path.join(
            self.args.output_dir, "trainer_state.json"))
        logger.info("Saving model checkpoint to %s", self.args.output_dir)

    def load_model(self):
        path = os.path.join(self.args.output_dir, 'checkpoint.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])

    def write_evaluation_result(self, out_file, results):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        out_file = self.args.output_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(
                key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()
