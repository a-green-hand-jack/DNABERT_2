# Pre-train bert-model for *Hierarchical transformer for genomics*

## 如何使用

一般来说只要运行下面的命令就行了：

```cmd
python dnabert-pretrain-k-mer.py --high_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-6/vocab.txt" \
                --low_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-3/vocab.txt" \
                --model_path="../tokenizer/tokenizer-config/dnabert-config/high-low-63-vocab.txt" \
                --data_path="../../Datasets/Human_genome/huixin/24_chromosomes-002.txt" \
                --output_dir="./dnabert-63-mer/results" \
                --logging_dir="./dnabert-63-mer/logs" \
                --num_train_epochs=2 \
                --per_device_train_batch_size=2
```
但是需要注意输入的文件夹之间的对应关系~~~

## 解释说明

> 因为这一部分十分关键，而且我做出了很多自己也不确定是不是正确的操作，所以我在这里记录我的思考过程。
>
> 这样，其他人在检查代码的时候也能更加轻松和明晰。

### `LineByLineTextDataset`

这里一开始有几个问题：
