## 数据预处理

### 极坐标转换

极坐标转换是数据预处理的第一步，输入和输出数据的极坐标转换都是可选的。此部分代码在 sortByPolar.py 中。除了极坐标转换外，此代码中同时还根据数据对成性只取了第四象限数据。输入是原始数据 train.h5 或 problem.h5。输出文件名表明了 input 或 output 以及 polar。

### 标准化

标准化是第二步，训练标准化模型的代码在 train_inputscaler.py 和 train_outputscaler.py 中。输入是极坐标转换后的数据，输出是标准化的model，Scaler_target 或 Scaler_feature。

### 降维

降维是第三步，训练PCA模型的代码在 feature_reducetrain.py 和 target_reducetrain.py 中。输入标准化后的数据，输出model PCA。对数据进行标准化，降维操作的代码在 feature_reduce.py 和 target_reduce.py 中。输入标准化和PCA模型，以及极坐标转换后的数据，输出整个预处理后的数据。

## 聚类

聚类的训练和数据处理在 cluster.py 中。聚类是可选操作。输入预处理后的数据，输出聚类标签和分类后的数据。

## 回归

模型的训练和数据预测分别在 train.py 和 predict.py 中。train.py 输入某一类的数据（或预处理后的数据），输出回归模型。predict.py 输入测试集数据并进行预测输出。

## 后处理

数据的还原和后处理在 inversPolar.py 中。包括坐标转换，PCA恢复，标准化恢复，还原四个象限。输入预测数据输出恢复后数据。

## 其他

如果进行了聚类，两类数据的合并在 merge.py 中。此操作主要是为了评估最终效果。

如果需要将数据可视化，代码在 plot.py 中。