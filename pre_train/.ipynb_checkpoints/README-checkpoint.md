# Pre-train bert-model for *Hierarchical transformer for genomics*

## 如何使用

一般来说只要运行下面的命令就行了：

```cmd
python dnabert-pretrain-k-mer.py --high_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-6/vocab.txt"   --low_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-3/vocab.txt"  --model_path="../tokenizer/tokenizer-config/dnabert-config/high-low-63-vocab.txt" --data_path="../../Datasets/Human_genome/huixin/24_chromosomes-002.txt" --output_dir="./dnabert-63-mer/results" --logging_dir="./dnabert-63-mer/logs"  --num_train_epochs=1 --per_device_train_batch_size=64
```

```cmd
python my-trainer.py --high_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-6/vocab.txt"   --low_model_path="../tokenizer/tokenizer-config/dnabert-config/bert-config-3/vocab.txt"  --model_path="../tokenizer/tokenizer-config/dnabert-config/high-low-63-vocab.txt" --data_path="../../Datasets/Human_genome/huixin/24_chromosomes-002.txt" --output_dir="./my-trainer/results" --logging_dir="./my-trainer/logs"  --num_train_epochs=1 --per_device_train_batch_size=1
```

但是需要注意输入的文件夹之间的对应关系~~~

## 解释说明

> 因为这一部分十分关键，而且我做出了很多自己也不确定是不是正确的操作，所以我在这里记录我的思考过程。
>
> 这样，其他人在检查代码的时候也能更加轻松和明晰。

### `LineByLineTextDataset`

> 这里一开始有几个问题与对应的解决方法：

1. 为了实现`k-mer`分词需要在 DNA sequence 中每间隔**k个碱基**增加一个*空格*；对应的增加了funcation:`insert_spaces`
2. 因为要使用两个不同的`tokenizer` 6-mer 和 3-mer ，所以对**同一段DNA sequence**分词后会得到两段tensor；对应的增加了funcation：`subtract_value_for_values`和`tokenize_sequence`
   1. 其中`tokenize_sequence`的目的是把两段`tensor`连接起来
   2. 其中`subtract_value_for_values`的目的是保证两个tokenizer中的token具有不同的id并且共用同样的special token 与其对应的 id
3. 因为需要和`Trainer`相适应，`__getitem__`返回的是`{"input_ids":high_tokenized_tensor}`

这里，我不太确定的是，`subtract_value_for_values`的定义是不是合适，这样实现token ids 的偏移会不会对model的表现产生什么不良的影响。

### `DataCollatorForMLM`

> 这里，是联系`LineByLineDataset`和`Trainer`的关键一步，也是在这里消耗了很大的精力。

1. 使用了`pad_sequence`保证了一个批次中的输入的tensor的长度都是一样的，因为我在`LineByLineDataset`中并没有使用`pad`；
2. 同时，增加了一个`if instances.size(1) > self.tokenizer.model_max_length: instances = instances[:, :self.tokenizer.model_max_length]`，用来实现截断；
3. 调用了`self.mask_tokens`，其实就是把`DataCollatorForLanguageModeling`中的这个函数抄了一遍

这里的担心在于`pad_sequence`是不是合适的，因为这样按batch经行pad可能会导致**不同的batch之间，transformer的输入具有不同的形状**。

但是，我有不愿意规定所有的输入具有相同的长度，比如规定为 512 tokens。因为很难实现两个tokenizer对同一段 DNA sequence 的tokenized 的输出的长度的和正好是 512，就算是近似的使用truncation和padding也不容易，特别是这样的操作往往意味着需要 **遍历很长的一段序列**。我感觉这样实在是对时间的浪费，所以我就是简单的逐行的去阅读 DNA sequence，并且进行tokenize。

自然，这样的方法就可能造成不同的batch之间的长度不一样。

### `MyTrainer`

直接使用的`Trainer`总是有错误，我猜想和`output = model(**processed_data)`中`output`的形式有关，不知道为啥，这里的`output['labels']=None`这似乎是不正常的，如果读者有解释请告诉我（~~球球了~~）

我在这里就是使用了`input['labels']`和`outputs`计算了`loss`而且为了更好的计算还定义了一个新的`LabelSmoother`，这个类本来是`transformers`里面的，我这里把它搞出来修正了一下。

### `LabelSmoother`

名为“修正”，其实就是复制，并且删除了一些判断语句。没啥好说的。

### `split_txt_file`

这个函数用来划分`traindataset`和`evaldataset`，可以控制划分适合两者的比例，这里我控制为0.8，也就是train:eval=8:2

### `load_and_convert_tokenizer`

用来加载`tokenizer`，但是目前的功能还不完全。后面可能还是需要修改，不过在`k-mer`的语境下能用了。

值得注意的是这里：

```python
dna_tokenizer = load_and_convert_tokenizer(args.model_path)
data_collator = DataCollatorForMLM(tokenizer=dna_tokenizer, mlm=True, mlm_probability=0.15)
```

这里，除了用于`tokenize`的`high_dna_tokenizer`与`low_dna_tokenizer`之外，我还增加了一个`dan_tokenizer`。其在`vocab.txt`上是前两者的叠加。但是我看它的作用似乎就是在`DataCollatorForMLM`中传递`self.tokenizer.pad_token_id`。

如果其功能真的只有这个，我会考虑删除它，因为有点麻烦，特别是我们要使用`BPE`的时候。

### 其他

> 剩下的都不是什么关键的函数了，在此略过（doge

## 后期的工作

### tokenizer

还是要优化这个`load_and_convert_tokenizer`函数，因为这里我为了适应`DNABER`中提供的[vocab](https://github.com/jerryji1993/DNABERT/tree/master/src/transformers/dnabert-config/bert-config-6)而使用了`BertTokenizer(vocab_file=load_path, do_lower_case=False)`。

但是一般通过`bpe`训练得到的tokenizer的保存，我的设定是保存为`transforkers`的标准形式，也就是使用`tokenizer.save_pretrained(save_floder)`。

后面还是需要和这个加以统一。

### model

这里为了图省事使用了`BertForPreTraining.from_pretrained("bert-base-uncased")`，不确定是否合适，可能还需要讨论和确认。
