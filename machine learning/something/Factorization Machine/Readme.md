## 因子分解机(Factorization Machine)
[算法介绍](https://blog.csdn.net/google19890102/article/details/45532745)，其核心思想是利用矩阵分解估计大型稀疏矩阵并求解问题。

* 因子分解机(Factorization Machine, FM)是由Steffen Rendle提出的一种基于矩阵分解的机器学习算法。对于因子分解机FM来说，最大的特点是对于稀疏的数据具有很好的学习能力。现实中稀疏的数据很多，例如作者所举的推荐系统的例子便是一个很直观的具有稀疏特点的例子。

* 因子分解机FM算法可以处理如下三类问题：
1、回归问题(Regression)
2、二分类问题(Binary Classification)
3、排序(Ranking)