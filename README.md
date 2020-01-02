# dogs-vs-cats
dogs-vs-cats-redux-kernels-edition

本项目是Kaggle上的一个图片分类项目Dogs vs. Cats，项目要解决的问题是一个计算机视觉领域的图像分类问题。该项目向我们提供了25000张带标签的猫狗照片作为训练集，以及12500张无标签猫狗照片作为测试集，通过训练一个猫狗分类模型，来预测这12500张无标签图是狗的概率。  

kaggle使用交叉熵loss值作为评估指标，值越小说明模型拟合的分布约接近真实分布，模型表现越好。交叉熵损失函数公式定义如下：  
![](picture/equation.svg)

主要思路：  
1、由于训练集存在非正常图片，首先要清洗数据，筛除异常图片
2、本项目主要使用pytorch、cv2、PIL等第三方库，通过迁移学习的方式创建CNN模型，实现图片分类器。  
	ResNet50和Inception_v3预训练模型作为特征提取器，合并两个输出特征，通过全连接层进行图片分类。  
	
**一、数据预处理  

1、识别图片文件名，整理成(path,label)的格式。  

2、数据清洗  
肉眼粗略查看训练集，发现存在一些这样的照片：  

非猫狗图片  
![image](picture/dog.4367.jpg)

背景复杂，人比狗多  
![image](picture/dog.6725.jpg)

巨大白边  
![image](picture/dog.9076.jpg)

一些背景复杂、猫狗占比过小、分辨率过小，或者根本就不是猫狗的图片混杂在训练集中。这会让模型学习到垃圾数据，需要先进行数据清洗，排除异常值。  
    
1、定义图片过滤器image_filter()，该函数有两个功能：  
①去除白边。巨大的白色边缘会导致裁剪时无法剪到完整的猫狗图片；  
②过滤掉像素小于12544的图片，过小的图片无法很好的表现一只猫狗。  


调用image_filter函数，将清洗后的训练集放入新文件夹./dataset/train_sets。  
清洗后剩下24803张图片，这个训练集足够大，损失197张图没有什么影响。  

**以上是第一次数据清洗

2、分割训练集和验证集  
为保证模型能够均衡学习猫狗特征，我们使用数量相等的猫狗图片来训练，猫和狗各9921张，共占原始训练集的80%，剩下4961张作为验证集。  
    
第二步，数据变换。  
使用torchvision的transforms.Compose()来组合图片变换函数，考虑到图片要分别输入两个模型，且两个模型要求的输入尺寸不一致(ResNet50：3*224*224，Inception_v3：3*299*299)，为保证输入的是同一张图的不同尺寸，变换组合中没有使用随机元素，只是用了CenterCrop。  
使用torch.utils.data.DataLoader()装载图片，我自己写了batch转换函数batch_transform，输入batch路径，输出batch Tensor:(batch, Channel, width, height)  
转化后的图片如下：  
    pic here
    
**二、创建模型  

1、模型  
定义混合模型类FusionNet，该类包含预训练CNN模型ResNet50和Inception_v3，因为预训练的参数已经很好，我们只进行finetune，冻结fully connected layer之前所有layer的参数，关闭Inception_v3的辅助分支，在ResNet50和Inception_v3两个模型之后添加自定义的fully connected layer：classifier=nn.Linear(4096,2)。这里有一个小trick，因为pytorch无法下载不包含fc层的预训练模型，我自定义了一个继承nn.Module的类FC(仅仅是为了包装成nn.Module子类)，用于替换ResNet50和Inception_v3的fc层。  
2、损失函数  
因为kaggle使用binary cross entropy作为评分函数(实际是计算loss)，我们选择torch.nn.CrossEntropyLoss()作为损失函数，以保证预测loss和kaggle一致  
3、优化器  
优化函数使用随机梯度下降算法，人为控制学习率，只更新全连接层的参数。  
optimizer=torch.optim.SGD(model.classifier.parameters(),lr=0.001,momentum=0.92)  

**三、训练模型  

训练过程中记录训练集和验证集的loss、accuracy，以便观察模型是否过拟合。

1、先用带杂质的训练集训练两轮epoch，由于训练集大部分数据仍属于正常值(正常的猫狗图片)，此时模型已学到相当量的猫狗特征，准确率达。
2、再用该训练集做一次预测，只计算loss，loss>0.43的样本预测概率不超过65%，十有八九是异常值，应该被清除，留下loss<=0.43的样本。  
**以上是第二次数据清洗  

3、使用第二次清洗过后的训练集重新训练新的模型(模型初始化)，在第?个epoch，模型准确率达到98%？？，平均loss：？？

**四、评价指标可视化

绘制loss和accuracy
    pic here
**五、预测

使用训练好的模型预测测试集，对输出使用Softmax函数，得到维度(batch,2)的Tensor，第二列(索引[:,1])便是图片为dog的概率。  
    
**六、对本项目的思考

模型相对于这个二分类问题来说有些复杂，数据拟合很快，也很容易过拟合，需要用非常小的学习率慢慢逼近最低loss。
模型之所以复杂是因为我一开始找错了问题方向，训练集中异常数据非常多，作为一个新手，刚开始我根本没有在意，总以为应该增加模型复杂度，来更好拟合数据，殊不知在某些任务中清洗数据可能比构建复杂模型更为重要。
