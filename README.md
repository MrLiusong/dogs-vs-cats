# dogs-vs-cats
dogs-vs-cats-redux-kernels-edition

    本项目使用迁移学习的方式创建CNN模型来参加Kaggle的dogs-vs-cats比赛项目，该项目向我们提供了25000张带标签的猫狗照片作为训练集，以及12500张猫狗照片作为测试集，训练一个猫狗分类模型，用于预测12500张无标签的图片。
    主要思路：
    本项目使用PIL、cv2、pytorch等第三方库来实现这个分类模型。采用ResNet50和Inception_v3预训练模型作为特征提取器，合并两个模型的特征输出，来训练一个全连接层。
    
    一、数据预处理
    识别图片标签，整理成(path,label)的格式。
    
    拿到训练集粗略看了一下，发现存在一些这样的照片
    pic here
    
    第一步，数据清洗。
    为防止rubbish in, rubbish out. 我们需要先进行数据清洗。
    1、定义一个图片过滤器image_filter，该函数有两个功能：
        去除白边，巨大的白色边缘会导致图片裁剪时无法剪到完整的猫狗图片；
        过滤掉像素小于12544的图片，过小的图片无法很好的表现一只猫狗。
    查看所有图片文件名，整理成文件路径，以便读取图片信息。
    调用image_filter函数，将清洗后的训练集放入新文件夹./dataset/train_sets，清洗过后剩下24803张图片，这个训练集足够大，可以忽略损失的197张图。
    
    2、分割训练集和验证集
    为保证模型能够学习到等量猫狗特征，我们将使用相同数量的猫狗图片作为训练集
    原始训练集的80%：sample_size=int(data_size*0.8*0.5)，猫狗各9921张，剩下4961张作为验证集。
    
    第二步，数据变换。
    使用torchvision的transforms.Compose()来组合图片变换函数，考虑到图片要分别输入两个模型，且两个模型的输入尺寸不一致(ResNet50：3*224*224，Inception_v3：3*299*299)，为保证输入的是同一张图片，变换组合中没有使用随机元素，只是用了CenterCrop。
    使用torch.utils.data.DataLoader()装载图片，我自己写了batch转换函数path2input，输入batch路径，输出适合模型的Tensor(batch, Channel, width, height)
    展示转化后的图片
    
    二、创建模型
    定义混合模型类FusionNet，其中包括预训练的模型ResNet50和Inception_v3，因为预训练的参数已经很好，我们只进行finetune，冻结fully connected layer之前的所有layer，关闭Inception_v3的辅助分支，在两个模型之后添加自定义的fully connected layer(4096,2)。
    因为kaggle使用binary cross entropy作为评分函数，我们用torch.nn.CrossEntropyLoss()作为损失函数，以保证预测loss和kaggle一致
    优化器使用随机梯度下降torch.optim.SGD()
    
    三、万事俱备，开始训练
    训练过程中记录训练集、验证集的loss和accuracy，以观察模型是否过拟合，
    
    四、绘制loss和accuracy
    
    五、使用训练好的模型预测测试集
    
    由于模型相对于这个二分类问题来说有些过于复杂，数据拟合很快，也很容易过拟合，需要用非常小的学习率慢慢逼近最佳点，在恰当的时机停止训练。
