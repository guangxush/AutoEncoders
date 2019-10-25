<h2 align="center">基于遗传算法做特征选择</h2>

### 运行环境
- python3.5+
- numpy
- pandas
- matplotlib
- sklearn
- LightGBM

### 使用手册

- 更换dataSet目录里面的数据集（数据集过大我只上传了一小部分）
    - train_feature.csv : 训练集
    - validate_feature.csv ： 验证集
    - 注：两个数据集内的维度要一致。且包含一个标签列，并命名为“target”
    
- 更换self.columns为train_feature.csv内的属性名，且第一个元素必须为标签名“target”,其它全为特征名称

- Genetic_algorithm.py众初始化种群initPopulation那里也要做相应修改

- 修改self.ga类参数(可选)

- 修改主函数里的群体个数和迭代次数(可选)
***

### 模型及评价指标
- 评价指标：auc     (可修改)
- 模型：LightGBM    (可修改)
***

### 结果
程序运行过程会打印出中间过程，最终会绘出迭代次数与最优个体适应图的[折线图](https://github.com/rogeroyer/feature_selection_GAAlgorithm/blob/master/result.jpg)以及打印出最有个体及其适应度。

***

### Attention

`经过两个不同比赛的尝试，评价指标分别是f1_score和auc，线上线下同增减的使用本算法才能收获一个好的结果。`
