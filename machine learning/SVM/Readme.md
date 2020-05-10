## 支持向量机(SVM)简介

### [基本介绍](https://zhuanlan.zhihu.com/p/31886934)

SVM学习的基本想法是求解能够正确划分训练数据集并且几何间隔最大的分离超平面。对于线性可分的数据集来说，这样的超平面有无穷多个，但是几何间隔最大的分离超平面却是唯一的。

其损失函数为：
$$ L(\mathbf{w},b) = \frac{1}{2}\left \| \mathbf{w}  \right \|^{2}+C \underset{i}{\sum}max(0,1-y_{i}(\mathbf{w} ^{T}\mathbf{x}_{i}+b)) $$

### LR与SVM的区别与联系

LR的损失函数是Cross entropy loss，svm的损失函数是Hinge loss，两个模型都属于线性分类器，而且性能相当。区别在于

* LR的输出具有自然的概率意义，而SVM的输出不具有概率意义；
* LR适合于大样本学习，SVM适合于小样本学习。

换用其他的Loss函数的话，SVM就不再是SVM了。正是因为Hinge Loss的零区域对应的正是非支持向量的普通样本，从而所有的普通样本都不参与最终超平面的决定，这才是支持向量机最大的优势所在，对训练样本数目的依赖大大减少，而且提高了训练效率。

## SVM原理推导
### 线性支持向量机
给定训练样本集 $D = \{(\mathbf{x}_{1}, y_{1}),  (\mathbf{x}_{2}, y_{2}), \cdots, (\mathbf{x}_{m}, y_{m})\}$, 在样本空间中，划分超平面可通过线性方程来描述：
$\mathbf{w}^{T} \mathbf{x} +b=0$.

假设超平面$(\mathbf{w}, b)$能将训练样本正确分类，即对于$(\mathbf{x}_{i}, y_{i}) \in D$, 若$y_{i} = +1$, 则有$\mathbf{w}^{T} \mathbf{x}_{i} + b > 0$; 若$y_{i} = -1$, 则有$\mathbf{w}^{T} \mathbf{x}_{i} + b < 0$.

令
$$
\left\{\begin{matrix}
\mathbf{w}^{T} \mathbf{x}_{i} + b \geq +1, y_{i} = +1 \\ 
\mathbf{w}^{T} \mathbf{x}_{i} + b \leq +1, y_{i} = -1 
\end{matrix}\right.
$$

最大化不同类别样本的划分平面间隔，其目标函数为：

$$
\max_{\mathbf{w},b} \frac{2}{||w||^{2}}\\
s.t.y_{i}(\mathbf{w}^{T} \mathbf{x}_{i} + b) \geq 1, i = 1,2,...,m.
$$

等价于
$$
\min_{\mathbf{w},b} \frac{||w||^{2}}{2}\\
s.t. y_{i}(\mathbf{w}^{T} \mathbf{x}_{i} + b) \geq 1, i = 1,2,...,m.
$$

### 对偶问题的转化
将上述凸优化问题转化为对偶问题：

$$
L(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}||\mathbf{w}||^{2} + \sum_{i=1}^{m}\alpha_{i}(1-y_{i}(\mathbf{w}^{T} \mathbf{x}_{i} + b )),
$$

其中 $\mathbf{\alpha}=(\alpha_{1};\alpha_{2};\cdots,\alpha_{m})$.令 $L(\mathbf{w},b,\mathbf{\alpha})$对$\mathbf{w}$和$b$的偏导数求零可得

$$
\mathbf{w} = \sum_{i=1}^{m}\alpha_{i}y_{i}\mathbf{x}_{i},\\
0 = \sum_{i=1}^{m}\alpha_{i}y_{i}
$$

于是得到对偶问题

$$
\begin{aligned}
\max_{\mathbf{\alpha}}&\sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\mathbf{x}_{i}^{T}\mathbf{x}_{j}\\
s.t. & \sum_{i=1}^{m}\alpha_{i}y_{i} = 0\\
& \alpha_{i} \geq 0, i=1,2,\cdots,m.
\end{aligned}
$$

解出 $\mathbf{\alpha}$ 后，求出 $\mathbf{w}$ 与 $b$ 即可得到模型

$$
\begin{aligned}
f(\mathbf{x}) &= \mathbf{w}^{T}\mathbf{x} + b
&= \sum_{i=1}^{m}\alpha_{i}y_{i}\mathbf{x}_{i}^{T}\mathbf{x} + b.
\end{aligned}
$$

需满足KKT条件如下：

$$
\left\{\begin{matrix}
\alpha_{i} \geq 0\\ 
y_{i} f(\mathbf{x}_{i}) -  1 \geq 0\\ 
\alpha_{i}(y_{i} f(\mathbf{x}_{i}) -  1)=0
\end{matrix}\right.
$$


若 $\alpha_{i} = 0$ ，则该样
本不会对 $f(x)$ 有任何影响;若$\alpha_{i} > 0$,
则必有$y_{i}f(\mathbf{x}_{i}) = 1$，所对应的样本点位于最大间隔边界上，是一个支持向量.

### 核空间（非线性）支持向量机
令 $\phi(\mathbf{x})$ 表示将 $\mathbf{x}$ 映射后的特征向量，于是， 在特征空间中的最大化间隔划分超平面所对应的模型应满足
$$
\min_{\mathbf{w},b} \frac{||w||^{2}}{2}\\
s.t. y_{i}(\mathbf{w}^{T} \phi(\mathbf{x})_{i} + b) \geq 1, i = 1,2,...,m.
$$
对偶问题

$$
\begin{aligned}
\max_{\mathbf{\alpha}}&\sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(\mathbf{x})_{i}^{T}\phi(\mathbf{x})_{j}\\
s.t. & \sum_{i=1}^{m}\alpha_{i}y_{i} = 0\\
& \alpha_{i} \geq 0, i=1,2,\cdots,m.
\end{aligned}
$$

解出 $\mathbf{\alpha}$ 后，求出 $\mathbf{w}$ 与 $b$ 即可得到模型

$$
\begin{aligned}
f(\mathbf{x}) &= \mathbf{w}^{T}\mathbf{x} + b\\
&= \sum_{i=1}^{m}\alpha_{i}y_{i}\phi(\mathbf{x}_{i})^{T}\phi(\mathbf{x}) + b\\
&= \sum_{i=1}^{m}\alpha_{i}y_{i}\kappa(\mathbf{x},\mathbf{x}_{i}) + b.
\end{aligned}
$$


### 软间隔与正则化 --> (核心是允许部分样本划分出错或越过分界面)

若采用hinge损失，则优化公式变成：

$$
\min_{\mathbf{w},b} \frac{1}{2}||\mathbf{w}||^{2} + C\sum_{i=1}^{m}max(0, 1-y_{i}(\mathbf{w}^T\mathbf{x}_{i}+b)).
$$

引入“松弛变量” $\xi_{i} \geq 0$,可转为：

$$
\min_{\mathbf{w},b} \frac{1}{2}||\mathbf{w}||^{2} + C\sum_{i=1}^{m} \xi_{i}
$$
$$
\begin{aligned}
s.t. &y_{i} (\mathbf{w}^{T}\mathbf{x}_{i}+b) \geq  1 - \xi_{i}\\ 
& \xi_{i} \geq 0, i=1,2,\cdots,m.
\end{aligned}
$$

通过拉格朗日乘子法得到拉格朗日函数：

$$
\begin{aligned}
L(\mathbf{w},b,\mathbf{\alpha}, \mathbf{\xi}, \mathbf{\mu}) = &\frac{1}{2}||\mathbf{w}^{2} + C\sum_{i=1}^{m}\xi_{i}\\
&+\sum_{i=1}^{m}\alpha_{i}(1-\xi_{i}-y_{i}(\mathbf{w}^{T}\mathbf{x}_{i}+b)) - \sum_{i=1}^{m}\mu_{i}\xi_{i}
\end{aligned}
$$

其中，$\alpha_{i} \geq 0, \mu_{i} \geq 0$ 是拉格朗日乘子。

令$L(\mathbf{w},b,\mathbf{\alpha}, \mathbf{\xi}, \mathbf{\mu})$对 $\mathbf{w},b,\xi_{i}$的偏导数求零可得

$
\mathbf{w} = \sum_{i=1}^{m}\alpha_{i}y_{i}\mathbf{x}_{i},\\
0 = \sum_{i=1}^{m}\alpha_{i}y_{i},\\
C = \alpha_{i} + \mu_{i}
$

代如方程得到对偶问题的优化函数如下：

$$
\begin{aligned}
\max_{\mathbf{\alpha}}&\sum_{i=1}^{m}\alpha_{i} - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(\mathbf{x})_{i}^{T}\phi(\mathbf{x})_{j}\\
s.t. & \sum_{i=1}^{m}\alpha_{i}y_{i} = 0\\
& 0 \leq \alpha_{i} \leq C, i=1,2,\cdots,m.
\end{aligned}
$$

对应的KKT条件如下：

$$
\left\{\begin{matrix}
\alpha_{i} \geq 0， \mu_{i}\geq 0\\ 
y_{i} f(\mathbf{x}_{i}) -  1 + \xi_{i} \geq 0\\ 
\alpha_{i}(y_{i} f(\mathbf{x}_{i}) -  1 + \xi_{i})=0\\
\xi_{i} \geq 0, \mu_{i}\xi_{i} = 0
\end{matrix}\right.
$$



### SMO算法
基本思路是先固定 $\alpha_{i}$ 外的其他参数，然后求 $\alpha_{i}$ 的极值。由于存在约束条件$\sum_{i=1}^{m}\alpha_{i}y_{i} = 0$ , 因此，每次选取两个变量$\alpha_{i}$ ，$\alpha_{j}$ ，并固定其他参数。不断迭代更新直至收敛。
* 选取变量的技巧（加速收敛）

1、第一个变量选取违反KTT条件最严重的变量；

2、第二个变量选与第一个变量差距最远的变量，更新速度更快。


# 问题分析
* 对偶问题之间怎么转换 -> 凸优化问题，将w与拉格朗日乘子 $\alpha$ 的角色求解变换，利用KKT条件,SMO算法取求解。