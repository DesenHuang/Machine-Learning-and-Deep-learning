[FM、FFM、DeepFM联系](https://zhuanlan.zhihu.com/p/61096338)

FM（Factorization Machines，因子分解机）最早由Steffen Rendle于2010年在ICDM上提出，它是一种通用的预测方法，在即使数据非常稀疏的情况下，依然能估计出可靠的参数进行预测。与传统的简单线性模型不同的是，因子分解机考虑了特征间的交叉，对所有嵌套变量交互进行建模（类似于SVM中的核函数），因此在推荐系统和计算广告领域关注的点击率CTR（click-through rate）和转化率CVR（conversion rate）两项指标上有着良好的表现。此外，FM的模型还具有可以用线性时间来计算，以及能够与许多先进的协同过滤方法（如Bias MF、svd++等）相融合等优点。


FM 施加的限制是要求二阶项系数矩阵是低秩的，能够分解为低秩矩阵的乘积.
