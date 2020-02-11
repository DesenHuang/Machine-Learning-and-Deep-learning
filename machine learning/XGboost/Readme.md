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
