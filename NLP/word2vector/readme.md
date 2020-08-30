## word2vector 有 一套embedding， 一套权重参数
* embedding参数将one-hot向量编码到特征空间
* 权重参数训练优化目标，如CBOW预测中心词，Skipgram预测上下文

有了IDF的定义，我们就可以计算某一个词的TF-IDF值了：

TF−IDF(x)=TF(x) * IDF(x)

其中TF(x)指词x在当前文本中的词频。


