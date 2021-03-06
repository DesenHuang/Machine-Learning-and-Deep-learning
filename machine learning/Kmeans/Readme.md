基本K-Means算法的思想很简单，事先确定常数K，常数K意味着最终的聚类类别数，首先随机选定初始点为质心，并通过计算每一个样本与质心之间的相似度(这里为欧式距离)，将样本点归到最相似的类中，接着，重新计算每个类的质心(即为类中心)，重复这样的过程，知道质心不再改变，最终就确定了每个样本所属的类别以及每个类的质心。由于每次都要计算所有的样本与每一个质心之间的相似度，故在大规模的数据集上，K-Means算法的收敛速度比较慢。
## 步骤：
* 1、初始化常数K，随机选取初始点为质心

* 2、重复计算以下3、4过程，直到质心不再改变

* 3、计算样本与每个质心之间的相似度，将样本归类到最相似的类中

* 4、重新计算质心

* 5、输出最终的质心以及每个类

## kmeans是GMM的一种特例，属于生成式模型；
[kmeans是hard EM，GMM是soft EM](https://zhuanlan.zhihu.com/p/71574416)

Kmeans实际上是混合高斯模型的特殊例子。

* EM算法针对带有隐变量的目标函数进行优化，分为两个迭代的step：

1、首先，E步做的工作是对隐变量z进行一个均值估计，此时p和q的kl散度为0，p和q在均值点出重合；

2、然后，固定隐变量z的均值点，进行模型参数最大化，寻找上界再利用上界出的模型参数重新估计隐变量z，反复迭代。

* 具体的，Kmeans算法的隐变量即样本的类别分布；
而GMM的隐变量为同样也是样本的类别分布，只不过每个高斯分布的组成比例可以不同。

KMeans算法的收敛性保证：https://blog.csdn.net/u010161630/article/details/52585764
