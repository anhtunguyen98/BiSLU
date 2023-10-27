from transformers import AutoTokenizer
from utils.dataloader import MyDataSet
from trainer import Trainer
from utils.utils import read_label
import os
import argparse

def main(args):

    print(args)

    intent_label_set = read_label(os.path.join(
        args.data_folder, 'intent_label.txt'))
    slot_label_set = read_label(os.path.join(
        args.data_folder, 'slot_label.txt'))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = MyDataSet(args=args,
                              folder=os.path.join(args.data_folder, 'train'),
                              tokenizer=tokenizer,
                              intent_label_set=intent_label_set,
                              slot_label_set=slot_label_set)
    dev_dataset = MyDataSet(args=args,
                            folder=os.path.join(args.data_folder, 'dev'),
                            tokenizer=tokenizer,
                            intent_label_set=intent_label_set,
                            slot_label_set=slot_label_set)

    test_dataset = MyDataSet(args=args,
                             folder=os.path.join(args.data_folder, 'test'),
                             tokenizer=tokenizer,
                             intent_label_set=intent_label_set,
                             slot_label_set=slot_label_set)

    trainer = Trainer(args=args,
                      tokenizer=tokenizer,
                      train_dataset=train_dataset,
                      dev_dataset=dev_dataset,
                      test_dataset=test_dataset,
                      intent_label_set=intent_label_set,
                      slot_label_set=slot_label_set)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Pre-trained model selected in the list: bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased...")

    parser.add_argument("--data_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--tuning_metric", default="mean_intent_slot",
                        type=str, help="Metrics to tune when training")

    parser.add_argument("--logging_steps", type=int,
                        default=200, help="Log every X updates steps.")

    parser.add_argument('--loss_coef_intent', default=0.5, type=float,
                        help="loss coef between intent and slot")

    parser.add_argument('--loss_coef_slot_scl', default=0.5, type=float,
                        help="loss coef between intent and slot")

    parser.add_argument('--loss_coef_intent_scl', default=0.5, type=float,
                        help="loss coef between intent and slot")


    parser.add_argument('--use_soft_slot', action='store_true',
                        help="Whether to use soft slot.")

    parser.add_argument('--use_scl', action='store_true',
                        help="Whether to use slot cnn.")

    parser.add_argument('--use_intent_context_attention', action='store_true',
                        help="Whether to intent slot attention")

    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help="dropout rate")
    
    parser.add_argument('--sd_loss_coef', default=0.5, type=float,
                        help="loss coef between intent and slot")

    parser.add_argument('--hidden_dim_ffw', default=300, type=int,
                        help="hidden dim of ffn start and ffn end")


    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )
    parser.add_argument('--use_sd', action='store_true',
                    help="Whether to use slot cnn.")

    args = parser.parse_args()
    main(args)
