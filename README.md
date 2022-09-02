# 2022 WAIC 黑客松蚂蚁财富赛道：AntSQL大规模金融语义解析中文Text-to-SQL挑战赛

[竞赛链接](https://tianchi.aliyun.com/competition/entrance/231716/introduction)

## 成绩

本项目所采用的方案在初赛中排30，正确率为74.5，虽然排名不高，但对入门SQL有一定参考价值。

因为疫情被隔离了未参加决赛，服了学校这些瓜皮领导，防疫过当了啊，最气人的是大爷可以随意在校内遛狗，学生禁止外出。

本人小白一枚，零基础参赛，代码写得不好，多多包涵

## 任务
将NLP语言转为机器可以理解的SQL语言

模型的输入为一个 Question，输出一个 SQL 结构，该 SQL 结构对应一条 SQL 语句。

其中 
- `sel` 为一个 list，代表 `SELECT` 语句所选取的列
- `agg` 为一个 list，与 `sel` 一一对应，表示对该列做哪个聚合操作，比如 sum, max, min 等
- `conds` 为一个 list，代表 `WHERE` 语句中的的一系列条件，每个条件是一个由 (条件列，条件运算符，条件值) 构成的三元组
- `cond_conn_op` 为一个 int，代表 `conds` 中各条件之间的并列关系，可以是 and 或者 or

## 参考

本方案受到了[基于Bert的NL2SQL模型：一个简明的Baseline](https://kexue.fm/archives/6771)这篇文章以及首届中文NL2SQL[第三名方案](https://github.com/beader/tianchi_nl2sql?spm=5176.21852664.0.0.14bf324eLiLVCn)的启发。

## 方案介绍

懒得画图了，不懂找我交流 ,分为两个模型，第一个模型预测`conds_op`, `conds_col`, `sel_col`;第二个模型预测`cond_conn_op`，`conds_value`

- Model 1

为了让模型更好理解列名与问题的关系，把所有列名添加至输入，本人新增第53列名‘无’

模型的输入的形式为： `CLS Question SEP CLS0 Column0 SEP0 CLS1 Column1 SEP1...CLS53 Column53 SEP53`

其中`CLS Question SEP`长128，不足以PAD补齐，总长度为512，不足以PAD补齐

BERT的输出CLS负责预测conds_op; 其他部分取出CLS，即`CLS0 CLS1...CLS53`经全连接层预测conds_col, sel_col

- Model 2 

模型的输入为问题+条件列名的形式：`CLS Question SEP cls conds_column SEP`, 长度为128

输出的CLS负责预测`cond_conn_op`，`conds_value`

其他部分的输出相当于命名实体识别问题（NER），我这里处理得不好，准确率不高主要是这一步的问题，
因为疫情所以没机会改进了，本项目用的首尾指针标注。我还试过beio，crf等效果都不好。
其实我下一步的想法是下载专门的NER预训练模型作为模型2，哎，可恶的疫情！

## 模型
ERNIE:https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

我选的是一个基于Bert的分类任务模型，可能这个模型不适合本项目，读者可选其他模型试试

## 使用说明

前言：这个运算量有点大，3090都跑不太动，尤其是模型1，batch size最多只支持24. 
所以放弃用自己电脑跑的想法吧

使用流程：
- 运行train_model1.py, 你会得到predict1
- 运行train_model2.py, 你会得到predict2
- 运行predict_final, 汇总predict1和predict2得到最终提交格式的文件
- 注意！：训练和预测时都需要注意配置相应文件的路径不要出错

其他文件说明
- Bert： 储存两个模型
- [crf：](https://github.com/kmkurn/pytorch-crf) 条件随机场，效果不好，未使用
- dataset: 处理数据，构建dataset
- utils: 一些函数
- draft: 草稿，没啥用
- data: 存放数据集
- predict_result: 存放预测结果
- pretain_model: 存放预训练模型


## 代码运行环境
* GPU RTX3090
* ubuntu 20.04.1
* cuda == 11.3
* python == 3.8.13 
* pytorch == 1.10.1 
* transformers==4.21.1   
* numpy==1.22.4
* sklearn
* jsonlines
* json

