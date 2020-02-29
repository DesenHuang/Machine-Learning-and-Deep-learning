## LR（从线性回归+sigmoid计算概率到似然函数进行模型参数估计，到损失计算）
* **二项逻辑斯蒂回归**是在线性回归模型的基础上，使用*sigmoid*函数，将线性模型 $z =  \mathbf{w}^{T}\mathbf{x}$ 
的结果压缩到[0,1] 之间
$$ P(y|z) = \frac{1}{1+e^{-z}} $$
得到样本x属于标注y的概率。针对二分类问题，其概率函数为
$$ P(y=1|x) = \frac{1}{1 + e^{-(w^{T}x+b)}} = \frac{e^{(w^{T}x+b)}}{1 + e^{(w^{T}x+b)}}$$  
$$ P(y=1|x) = 1 - P(y=1|x) =\frac{1}{1 + e^{(w^{T}x+b)}}$$

* **事件的几率**是指该事件发生的概率与该事件不发生的概率的比值。如果事件发生的概率是*p*，那么该事件的几率是$\frac{p}{1-p}$，该事件的对数几率(log odds)或logit函数是
$$ logit(p) = log\frac{p}{1-p} $$

为方便描述，我们扩充向量*w,x*,令$\mathbf{w} = (\mathbf{w}^{(1)},\mathbf{w}^{(2)},\dots,\mathbf{w}^{(n)},b)$, $\mathbf{x} = (\mathbf{x}^{(1)},\mathbf{x}^{(2)},\dots,\mathbf{x}^{(n)},1)$

--> 对于逻辑斯地回归而言，输出$y=1$的对数几率是输入x的线性函数。即：
$$ log\frac{P(y=1|x)}{1-P(Y=1|x)} = w^{T}x + b$$
* 其训练分类器的交叉熵损失函数实际上就是一个似然。

**模型参数估计:** 在逻辑斯蒂回归模型学习时，对于给定的训练数据集$T={(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})}$，其中，$x_{i}\in \mathbb{R}^{n}$, $y_{i} \in \{0,1\}$ ,应用极大似然估计法估计模型参数，从而得到逻辑斯蒂回归模型。

设：$P(y=1|\mathbf{x}) = \mathbf{\pi(x)}$ ，$P(y=0|\mathbf(x) = 1 - \mathbf{\pi(x)})$

似然函数为：
$$ \prod_{i=1}^{N}[\mathbf{\pi}(\mathbf{x}_{i})]^{y_{i}}[1-\mathbf(\pi)(\mathbf{x})]^{1-y_{i}}$$

对数似然函数为：
$$
\begin{aligned}
L(\mathbf{w}) &= \sum_{i=1}^{N}y_{i}log \mathbf{\pi}(\mathbf{x}_{i})+(1-y_{i})log(1-\mathbf{\pi}(\mathbf{x}_{i}))\\
& =\sum_{i=1}^{N} \left [y_{i}log \frac{\mathbf{\pi}(\mathbf{x}_{i})}{1-\mathbf{\pi}(\mathbf{x}_{i})} + log(1-\mathbf{\pi}(\mathbf{x}_{i})) \right ]\\
& =\sum_{i=1}^{N}[y_{i}(\mathbf{w}^{T}\mathbf{x}_{i}) - log(1+e^{\mathbf{w}^{T} \mathbf{x}_{i}})]
\end{aligned}
$$

求似然的极大值采用梯度上升法，设步长为$\alpha$,则迭代得到的新的权重参数为：$\mathbf{w} = \mathbf{w} + \alpha \cdot \bigtriangledown L(\mathbf{w})$, 其中 

$$
\begin{aligned}
\bigtriangledown L(\mathbf{w}) = \frac{\partial L(\mathbf{w})}{\partial{\mathbf{w}}} &= \sum_{i=1}^{N} [y_{i}\mathbf{x}_{i} - \frac{\mathbf{x}_{i}e^{\mathbf{w}^{T}\mathbf{x}_{i}}}{1+e^{\mathbf{w}^{T}\mathbf{x}_{i}}}]\\
&= \sum_{i=1}^{N} \mathbf{x}_{i}(y_{i}-\mathbf{\pi}({\mathbf{x}_{i}})) = \mathbf{x} \cdot \mathbf{error}
\end{aligned}
$$


<!-- $$
\begin{aligned}
x ={}& a+b+c+{} \\
&d+e+f+g
\end{aligned}
$$ -->

### 拓展到多分类
* 多项逻辑斯蒂回归 <-> Softmax
* [Softmax推导](https://zhuanlan.zhihu.com/p/25723112)

### Reference
* 李航，统计学习方法， 2012