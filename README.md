# Hierarchical transformer for genomics

这里包括以下内容： 

1. 如何搭建环境与运行程序
2. 现存的问题
3. 下一阶段的工作




## Contents

<!-- TOC -->

- [Hierarchical transformer for genomics](#hierarchical-transformer-for-genomics)
  - [Contents](#contents)
  - [Update (2024/03/12)](#update-20240312)
  - [搭建环境与运行程序](#搭建环境与运行程序)
    - [搭建conda环境](#搭建conda环境)
    - [运行程序](#运行程序)
  - [现在的问题](#现在的问题)
  - [近期的规划（2024-03-12）](#近期的规划2024-03-12)

<!-- /TOC -->


## Update (2024/03/12)

大致实现了在大样本上的分批次的`pre-train`，使用的`model`是`DNABERT-V2`。采用的`dataset`是`whole human genome`。



## 搭建环境与运行程序

### 搭建conda环境

自动化环境建立

```cmd
# clone 
git clone https://github.com/a-green-hand-jack/DNABERT_2.git
# conda create
conda env create -f environment.yml
```

如果这样不成功，也可以手动处理

```cmd
# create and activate virtual python environment
conda create -n dna python=3.8
conda activate dna
# clone 
git clone https://github.com/a-green-hand-jack/DNABERT_2.git
# install
pip install tokenizers==0.12.1
pip install transformers=4.29.2
```

目前来看`tokenizers`和`transformers`的版本比较关键，其他的库与之匹配即可。

### 运行程序

程序运行分为下面几个步骤：

1. 下载数据
2. 数据预处理
3. `tokenizer`的训练
4. `DNEBERT-V2`的训练

> 具体的运行步骤请进入对应文件夹中查阅！！！

## 现在的问题

> 有几个需要确认的问题，不确定会不会对`model`效果产生影响。


1. 数据预处理的分行
   1. 事实上把不同染色体上的序列混合在一起了，比如某一行中的数据很可能跨越了两个染色体。虽然影响应该不会太大，但是把原来不连续的两个序列放在一起总是叫人不安
2. 分行、分块的训练`tokenizer`，这样得到的`BPE`字典可靠吗？
   1. 这样做是受到cpu内存大小的限制迫不得已
   2. 但是这样得到的字典不一定可靠
   3. 希望有其他的可以减小cpu消耗但是又能保证字典准确的方法
   4. 或许一个可行的方式是：**逐染色体进行分词**，不过我没试过
3. 分行训练`model`
   1. 问题和`tokenizer`是类似的，不过也是`bert`的共同问题，就是输入的序列长度是不是有点**短**
   2. 而且这样输入`model`的做完分词后的`tensor`的长度每一个`batch`之间不一定一样的，会不会产生影响
   3. 这样每次输入`model`的碱基长度没有办法灵活的调整，感觉很低效
   4. 我期望的其实是可以在不使用`padding`和`truncation`的情况下控制每一次`token`结束后得到的输入到`model`的`tensor`的长度是一样的，不知道是否合理


## 近期的规划（2024-03-12）

> 大概写一下我的设想

1. 实现两个`toenizer`的共同分词
   1. 或许有两个技术路径：
      1. 利用用两个分词器分词得到的字典的并集，对一个片段进行分词（不过感觉这样会对**长**的有利，记得做BPE分词的时候是**贪心**的）
      2. 两个分词器分别对一段序列做分词，得到两个`tensor`为\(A\)和\(B\)；它们很可能会有不同的长度，然后把他们直接**连接**起来，得到一个新的张量\(H\)；直接把\(H\)送入`model`中
   2. 如果采用第二种方法，我觉得甚至可以尝试多种规模的`tokenizer`，比如同时用3个`tokenizer`
2. 可以尝试把[DNABERT-V2](https://arxiv.org/abs/2306.15006)和[GENA-LM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v2)里面的技术都用一下
   1. Flash Attention
   2. Attention with linear biases
   3. Sparse Self Attention
   4. Transformer-XL
   5. 但是这样也可能导致自己的创新点不够突出





