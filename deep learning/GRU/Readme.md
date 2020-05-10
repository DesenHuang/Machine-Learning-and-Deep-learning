Gru由重置门和更新门组成,其输入为前一时刻隐藏层的输出和当前的输入,输出为下一时刻隐藏层的信息。重置门用来计算候选隐藏层的输出,其作用是控制保留多少前一时刻的隐藏层。更新门的作用是控制加入多少候选隐藏层的输出信息,从而得到当前隐藏层的输出。

![avater](GRU.png)

GRU算法出自这篇文章："Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"


* 对比LSTM
![avater](../LSTM/LSTM.png)

GRU少了一个output gate，其中的z重置门相当于LSTM中的遗忘门.