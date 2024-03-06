# Prepare dataset for *Hierarchical transformer for genomics*

## 下载数据

最原始的数据在[GENcode-human](https://www.gencodegenes.org/human/)中，具体的讲是这个**GRCh38.p14**。不过也可以直接从[这个链接](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.p14.genome.fa.gz)中下载文件。

## 数据预处理

### 方法一

> 这个方法截止到2024-03-17就已经废止了，放在这里只是为了以防万一😀

进入`dataset`文件夹，运行`humanGenome.ipynb`，注意修改文件路径与保存路径；得到了对应的`csv`文件。

同样在`dataset`文件夹中，进入`visual.py`文件，修改文件路径、规定每一行的长度，运行程序；得到了对应的`txt`文件。

### 方法二

> 这个方法来源于[GENA_LM](https://github.com/AIRI-Institute/GENA_LM/blob/main/src/gena_lm/genome_tools/create_corpus.py)

进入`dataset`文件夹，运行下面的命令，注意要使用对应的`gz`文件路径：

```cmed
python create_corpus.py --input-file "../../Datasets/Human_genome/huixin/GRCh38.p14.genome.fa.gz" --output-dir "../../Datasets/Human_genome/huixin/create_corpus_norc" --rc-flag "no_rc" 
```

这里的`--rc-flag`可以控制在分割基因组的时候要不要考虑其**反向序列**，一般来说，为了节约空间可以不考虑。

如果**只**希望考虑**反向序列**可以运行下面的命令：

```cmd
python create_corpus.py --input-file "../../Datasets/Human_genome/huixin/GRCh38.p14.genome.fa.gz" --output-dir "../../Datasets/Human_genome/huixin/create_corpus_norc" --rc-flag "use_rc" 
```

进一步的，如果希望同时考虑正向和反向序列，可以使用下面的命令行：

```cmd
python create_corpus.py --input-file "../../Datasets/Human_genome/huixin/GRCh38.p14.genome.fa.gz" --output-dir "../../Datasets/Human_genome/huixin/create_corpus_norc" --rc-flag "both_rc" 
```