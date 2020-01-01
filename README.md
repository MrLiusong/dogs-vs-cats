# dogs-vs-cats
dogs-vs-cats-redux-kernels-edition

本项目使用迁移学习的方式创建CNN模型来完成Kaggle的dogs-vs-cats比赛项目，该项目向我们提供了25000张带标签的猫狗照片作为训练集，以及12500张无标签猫狗照片作为测试集，通过训练一个猫狗分类模型，来预测这12500张无标签图。
主要思路：
使用pytorch、cv2、PIL等第三方库来实现这个分类模型。采用ResNet50和Inception_v3预训练模型作为特征提取器，合并两个模型的特征输出，用于训练一个全连接层。
    
一、数据预处理

1、识别图片标签，整理成(path,label)的格式。

2、数据清洗
拿到训练集肉眼粗略看了一下，发现训练集存在一些这样的照片

非猫狗图片  
![](picture/dog.4367.jpg)

背景复杂，人比狗多  
![](picture/dog.6725.jpg)

巨大白边  
![](picture/dog.9076.jpg)

一些背景复杂、猫狗占比过小、分辨率过小，或者根本就不是猫狗的图片混杂在训练集中。为防止模型学习到垃圾数据，需要先进行数据清洗，排除训练集异常值。
    
1、定义一个图片过滤器image_filter()，该函数有两个功能：
去除白边。巨大的白色边缘会导致裁剪时无法剪到完整的猫狗图片；
过滤掉像素小于12544的图片，过小的图片无法很好的表现一只猫狗。


调用image_filter函数，将清洗后的训练集放入新文件夹./dataset/train_sets。
完成后剩下24803张图片，这个训练集足够大，可以忽略损失的197张图。
    
2、分割训练集和验证集
为保证模型能够均衡学习到猫狗特征，我们将使用数量相等的猫狗图片作为训练集，猫狗各9921张，一共占原始训练集的80%，剩下4961张作为验证集。
    
第二步，数据变换。
使用torchvision的transforms.Compose()来组合图片变换函数，考虑到图片要分别输入两个模型，且两个模型要求的输入尺寸不一致(ResNet50：3*224*224，Inception_v3：3*299*299)，为保证输入的是同一张图的不同尺寸，变换组合中没有使用随机元素，只是用了CenterCrop。  
使用torch.utils.data.DataLoader()装载图片，我自己写了batch转换函数batch_transform，输入batch路径，输出batch Tensor:(batch, Channel, width, height)
转化后的图片如下：
    pic here
    
二、创建模型
定义混合模型类FusionNet，该类包含预训练模型ResNet50和Inception_v3，因为预训练的参数已经很好，我们只finetune最后的一层分类器(classifier)，冻结fully connected layer之前所有的layer，关闭Inception_v3的辅助分支，在ResNet50和Inception_v3两个模型之后添加自定义的fully connected layer：classifier=nn.Linear(4096,2)。这里有一个小trick，因为pytorch无法下载不包含fc层的预训练模型，我自定义了一个继承nn.Module的类FC(其实这个类什么也干)，用于替换ResNet50和Inception_v3的fc层。
因为kaggle使用binary cross entropy作为评分函数，我们用torch.nn.CrossEntropyLoss()作为损失函数，以保证预测loss和kaggle一致
优化器使用随机梯度下降，人为控制学习率。
optimizer=torch.optim.SGD(model.classifier.parameters(),lr=0.001,momentum=0.92)
反向传播只更新最后全连接层的参数
    
三、万事俱备，开始训练
训练过程中记录训练集和验证集的loss、accuracy，以便观察模型是否过拟合，
    
四、绘制loss和accuracy
    
五、使用训练好的模型预测测试集
    
由于模型相对于这个二分类问题来说有些过于复杂，数据拟合很快，也很容易过拟合，需要用非常小的学习率慢慢逼近最佳点，并在恰当的时机停止训练。
训练集中的异常值非常多，不止我列出的这几种，还有改进的空间。
