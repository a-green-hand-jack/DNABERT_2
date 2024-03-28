import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
import glob
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path


def main(args):
    # paths = [str(x) for x in Path('/home/zhihan/dnabert_xl/splits').glob('**/*.fa')]
    paths = ["/home/user/local-private-zhihan/data/DNABERT-2/tokenizer/train.txt"]
    postfix = "_multi"

    vocab_size = args.vocab_size
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size = vocab_size, min_frequency=2)

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(paths, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    print("train finish")
    
    tokenizer_dir = args.tokenizer_dir + str(vocab_size) + postfix
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    # generate and save tokenizer config
    tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast", 
                        "unk_token": "[UNK]",
                        "cls_token": "[CLS]",
                        "sep_token": "[SEP]",
                        "pad_token": "[PAD]",
                        "mask_token": "[MASK]"}
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    # generate and save model config
    with open(os.path.join("data", "config.json"), "r") as f:
        model_config = json.load(f)
    model_config['vocab_size'] = vocab_size
    with open(os.path.join(tokenizer_dir, "config.json"), "w") as f:
        json.dump(model_config, f)

    print("tokenizer saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizers")
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()
    main(args)
