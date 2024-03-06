# Train tokenizer for *Hierarchical transformer for genomics*

## 训练分词器

当我们准备好用于训练分词器的数据后就可以进行分词了；如果cpu内存足够可以一次性训练出完整的`tokenizer`，这里给出一个实例：

```cmd
python dan-bert-tokenizer-trainer.py --input_dir "../sample_data" --output_dir "./tokenizer-config/gena_lm_tokenizer" --vocab-size 32768 --tokenizer_name "24_chromosomes-002" --output_dir_standard "./tokenizer-config/gena_lm_tokenizer/standard" 
```