## map-reduce 原理，怎么操作
Spark是UC Berkeley AMP lab (加州大学伯克利分校的AMP实验室)所开源的类Hadoop MapReduce的通用并行框架，Spark，拥有Hadoop MapReduce所具有的优点;但不同于MapReduce的是Job中间输出结果可以保存在内存中，从而不再需要读写HDFS，因此Spark能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。

Spark 是一种与 Hadoop 相似的开源集群计算环境，但是两者之间还存在一些不同之处，这些有用的不同之处使 Spark 在某些工作负载方面表现得更加优越，换句话说，Spark 启用了内存分布数据集，除了能够提供交互式查询外，它还可以优化迭代工作负载。

1、pyspark的高效使用简单指南。https://blog.csdn.net/cymy001/article/details/78483723
2、RDD(Resilient Distributed Dataset) 叫着 弹性分布式数据集 ，是Spark 中最基本的抽象，它代表一个不可变、可分区、里面元素可以并行计算的集合。

RDD 具有数据流模型特点：自动容错、位置感知性调度和可伸缩。

3、为什么需要 Yarn？
Yarn 的全称是 Yet Anther Resource Negotiator（另一种资源协商者）。它作为 Hadoop 的一个组件，官方对它的定义是一个工作调度和集群资源管理的框架。
