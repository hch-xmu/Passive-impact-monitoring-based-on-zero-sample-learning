## Data Description

​        为了验证所提出的方法在铝制结构上的有效性，首先选取铝板进行冲击实验。（编号）铝板的截面厚度为2mm，并对其两侧使用（结构）进行支撑。为了防止边界反射等影响因素影响信号的质量，冲击的区域划定在铝板的侧中央，大小为20cm*20cm。更进一步，我们将冲击区域划分为16个子区域，每个子区域大小为5cm*5cm。4个PZT传感器粘贴在铝板表面，分别位于划定的冲击区域的四个边角，两两传感器之间的距离为20cm。

​        通过搭建被动式冲击监测系统来实现数据的实时采集，该系统主要包括传感器模块和数据采集模块。当冲击事件发生时，通过结构上的传感器接收结构的冲击振动响应，将结构的振动响应转换成电压信号，再通过数据采集模块内部集成的调理模块对电压信号进行调理，然后通过数据采集模块将信号进行模拟信号到数字信号的转换，最后通过采集模块对信号进行实时采集。本文使用的数据采集模块是东华DH6523。为了保证采集的数据准确且不失真，在实验中采样频率设置为f_s=125kHz，单次冲击采集时间为2s。

​        考虑到实际应用过程中，在采集数据的过程中既进行冲击的过程中应该保证不对板材产生太大的冲击损伤，这需要我们在保证数据集的大小能够满足训练要求的基础上尽量减少冲击次数。由于使用类似橡胶材质对板材进行多次冲击而不产生损伤，其他刚性冲击会对板材造成一定影响。故本文设定了两种不同的冲击条件：一种是使用橡胶球冲击板材，其在迁移学习过程中被用作为源数据集；另一种是使用铁球冲击板材，其在迁移学习过程中被用作为目标数据集。源数据集的冲击次数设定为20次，其中17次冲击作为预训练过程中的训练集，另外3次作为测试集。为了讨论迁移学习可以实现使用很少的目标数据集实现高精度的冲击识别，目标数据集的训练集的冲击次数设定为1、2、3、4、5、6和7次，以及外加3次作为测试集。数据集具体情况见表1。

| Name | Total No.of Signals | No. of Sensors | Training Data | Test Data | Classes | Signal per Classes |
| ---- | ------------------- | -------------- | ------------- | --------- | ------- | ------------------ |
| TG1  | 92                  | 4              | 64            | 192       | 16      | 4                  |
| TG2  | 183                 | 4              | 128           | 192       | 16      | 8                  |
| TG3  | 274                 | 4              | 192           | 192       | 16      | 12                 |
| TG4  | 366                 | 4              | 256           | 192       | 16      | 16                 |
| TG5  | 457                 | 4              | 320           | 192       | 16      | 20                 |
| TG6  | 549                 | 4              | 384           | 192       | 16      | 24                 |
| TG7  | 640                 | 4              | 448           | 192       | 16      | 28                 |
| XJ20 | 1280                | 4              | 896           | 192       | 16      | 80                 |

![image-20220224161548232](C:\Users\22809\AppData\Roaming\Typora\typora-user-images\image-20220224161548232.png)

## Data preprocessing

​        触发阈值电压是系统监测到的信号输入大于该阈值才认定产生了冲击时间，这里通过调试将阈值电压设定为满量程的10%，此时能够保证系统在较小的误触发率的同时灵敏地捕捉到冲击信号。触发点前设定采集0.5s，触发点后设定采集1.5s，这样充分保证了采集到的冲击信号的完整性。图1是我们采集到的信号。
触发点前以及触发点后的采样个数也即截取的传感器信号时间窗宽度，主要用于确认冲击信号的完整性。如果设置的过小，会导致丢失部分冲击信号，设置过长会导致信号数据量增大，其会直接影响传输和储存速度以及模型的训练速度和精度。

![image-20220224161658009](C:\Users\22809\AppData\Roaming\Typora\typora-user-images\image-20220224161658009.png)

## 基于卷积神经网络的深度迁移学习

​        在所提出的方法中，使用隐藏层提取信号特征，最后在分类层使用Softmax函数对从隐藏层中提取的高级特征进行分类。对于深度神经网络，在使用源数据集预训练好的可迁移网络基础上进行重新使用目标数据集重新训练Softmax层的参数来完成不同的识别任务被证明是有效的[5]。在实时的被动式冲击监测研究中，该模型要求不但能实现对新冲击的高精度识别，还能在新冲击的较少情况下实现高精度识别，通过这种方法可以减少所需要采集的目标数据集以降低对结构体的损伤和采集数据的人力成本。
​        基于卷积神经网络的深度迁移学习的冲击识别分为以下**四个步骤**。详细过程如图示。

**步骤1**.**搭建基础模型**。此步骤搭建基础的卷积神经网络模型，选用了在图像识别领域取得较好效果的全卷积神经网络（FCN）作为基础模型。
**步骤2**.预训练以及优化基础模型。此步骤使用来自源数据集的冲击数据和相应标签来预训练 CNN 模型。‎在训练过程中通过调节超参数、参数等来不断优化基础模型，得到识别效果最优的新模型。通过该步骤能得到识别效果最好的预训练模型。
**步骤3**.**知识迁移**。将预训练模型的知识学习能力迁移到目标任务模型，其中目标任务模型是针对新任务既用于新冲击识别任务的新模型。其通过将预训练模型的参数移植到目标任务模型中实现知识迁移。
**步骤4**.微调目标任务模型。此步骤使用来自于目标数据集的冲击数据和相应标签来微调目标任务模型,在微调过程中通过不断调节参数和超参数来优化目标任务模型，实现针对目标任务的最好识别效果。

​        微调目标任务模型有两种方法：1）冻结部分隐藏层的参数，其他未冻结的参数依旧能够实现更新 2）不对任何隐藏层进行冻结，所有参数都能实现更新。

​        **冻结部分隐藏层：**该方法先将预训练模型的参数移植到目标模型，再冻结部分目标模型隐藏层的参数，其他未冻结的参数依旧能够实现更新。前几层隐藏层提取一般特征（与问题无关），后几层隐藏层提取特定特征（取决于问题）。通常，如果数据集较小且参数数量较多，则会冻结更多隐藏层以避免过度拟合。相比之下，如果数据集很大，参数数量较少，则可以通过为新任务训练更多层来改进模型，因为过拟合不是问题。为此，文中将通过讨论了冻结层数对模型性能的影响，找到最佳的冻结层数。

​        **移植的隐藏层其参数能进行更新：**该方法先移植需要进行参数移植预训练模型前几个卷积层的参数到目标模型，其他隐藏层的参数进行随机初始化，所有隐藏层包括输出设置为可更新既不冻结各层的参数，并随后使用目标数据集对模型进行训练。

