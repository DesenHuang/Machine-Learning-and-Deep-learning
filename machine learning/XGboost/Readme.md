## XGBoost
[项目地址](https://github.com/dmlc/xgboost)
[学习介绍](https://www.jianshu.com/p/7e0e2d66b3d4)

* XGBoost 所应用的算法就是 gradient boosting decision tree，既可以用于分类也可以用于回归问题中。

Gradient boosting 是 boosting 的其中一种方法

所谓 Boosting ，就是将弱分离器 f_i(x) 组合起来形成强分类器 F(x) 的一种方法。

所以 Boosting 有三个要素：

① A loss function to be optimized：
例如分类问题中用 cross entropy，回归问题用 mean squared error。

② A weak learner to make predictions：
例如决策树。

③ An additive model：
将多个弱学习器累加起来组成强学习器，进而使目标损失函数达到极小。


* 具体地，GDBT在函数空间中利用梯度下降法进行优化而XGB在函数空间中使用了牛顿法进行优化。即GDBT在优化中使用了一阶导数信息,而XGB对损失函数进行了二阶泰勒展开,用到了一阶和二阶倒数信息。XGB在损失函数中加入了正则项(树叶子节点个数,每个叶子节点上输出score的L2模平方和。对于缺失的样本,XGB可以自动学习出它的分裂方向。GDBT的节点分裂方式使用的是gini系数,XGB通过优化推导出分裂前后的增益来选择分裂节点。XGB在处理每个特征列时可以做到并行。