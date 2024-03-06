import itertools
from pathlib import Path
from typing import Optional
import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

DEFAULT_VOCAB_SIZE = 32_000


def create_bpe_tokenizer(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "BPE",
    limit_files: Optional[int] = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> str:
    """
    Train basic BPE tokenizer without preprocessing with BERT-like special tokens

    Reads all .txt files in given directory, saves
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    files = input_dir.glob("*.txt")
    if limit_files is not None or limit_files != 0:
        files = itertools.islice(files, 0, limit_files)

    print("开始训练tokenizer！！！")
    tokenizer.train([str(file) for file in files], trainer)
    print("tokenizer训练结束了！！！")
    print(f"保存tokenizer在{output_dir}")

    # 创建目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 tokenizer 到目录中
    # tokenizer.save_model(str(output_dir))
    tokenizer.save(str((output_dir / f"{tokenizer_name}.json").resolve()))
    
    return str((output_dir / f"{tokenizer_name}.json").resolve())

def load_bpe_tokenizer_save(tokenizer_path: str, save_floder:str) -> PreTrainedTokenizerFast:
        """
        Load BPE tokenizer from the saved JSON file
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # output_dir =save_floder + "/transformers-standard"
        # tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_floder)
        return tokenizer

@click.command()
@click.option("--input_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--output_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--output_dir_standard", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--limit-files", type=int, )
@click.option("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, show_default=True)
@click.option("--tokenizer_name", default="BPE", show_default=True)
def cli(input_dir, output_dir,output_dir_standard, limit_files, vocab_size, tokenizer_name):
    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = create_bpe_tokenizer(
        input_dir=input_dir,
        output_dir=output_dir,
        limit_files=limit_files,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
    )
 # 替换为你保存的json文件的路径
    loaded_tokenizer = load_bpe_tokenizer_save(tokenizer_path, output_dir_standard)
    


if __name__ == "__main__":

    """
    配合的提供一个实例：与./dataset/create_corpus.py对应
    python dan-bert-tokenizer-trainer.py --input_dir "../../Datasets/Human_genome/huixin/create_corpus" --output_dir "./tokenizer-config/gena_lm_tokenizer" --vocab-size 32000 --tokenizer_name "high_level" --output_dir_standard "./tokenizer-config/gena_lm_tokenizer/standard" 
    
    提供一个小的测试实例
    python dan-bert-tokenizer-trainer.py --input_dir "../sample_data" --output_dir "./tokenizer-config/gena_lm_tokenizer_small" --vocab-size 100 --tokenizer_name "small" --output_dir_standard "./tokenizer-config/gena_lm_tokenizer_small/standard" 
    """
    save_path = cli()
    
