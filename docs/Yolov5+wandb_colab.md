# Yolov5 and Colab

## 一、简介

YOLOv5 可以在以下任何经过验证的最新环境中运行（所有依赖项包括[CUDA](https://developer.nvidia.com/cuda) / [CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/)和[PyTorch](https://pytorch.org/)预安装）：

- 带有免费 GPU 的**Google Colab Notebook**：[![在 Colab 中打开](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)
- 带免费 GPU 的**Kaggle 笔记本**：[https](https://www.kaggle.com/ultralytics/yolov5) : [//www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **谷歌云**深度学习虚拟机。请参阅[GCP 快速入门指南](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **Docker 镜像** https://hub.docker.com/r/ultralytics/yolov5

## 二、yolov5的配置文件`yaml`

* YAML(YAML Ain`t Markup language)文件，它不是一个标记语言。配置文件有xml、properties等，但YAML是以数据为中心，更适合做配置文件。
* YAML的语法和其他高级语言类似，并且可以简单表达清单、散列表，标量等数据形态。
* 它使用空白符号缩进和大量依赖外观的特色，特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲。yaml介绍
* 大小写敏感；缩进不允许使用tab，只允许空格；缩进的空格数不重要，只要相同层级的元素左对齐即可；’#'表示注释；使用缩进表示层级关系。

Yolov5的配置文件为`yaml`类型，yolov5*.yaml文件通过yolo.py解析文件配置模型的网络结构。yaml文件配置网络的好处是十分的方便不需要像Yolov3的config设置网络一样进行叠加，只需要在yaml配置文件中的参数进行修改即可。为了更好的了解和修改yolov5的配置文件下面以yolov5s.yaml文件为例介绍网络配置文件的参数。其中depth_multiple控制网络的深度，width_multiple控制网络的宽度。

这样做的好处是可以通过这两个参数控制网络的宽度和深度，实现不同大小不同复杂度的模型设计，但是这种写法缺点是不再能直接采用第三方工具例如netron进行网络模型可视化。

### yaml里有什么

这里以[**Yolov5s.yaml**](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml)为例，共分为四个部分。yolov5提供了s、m、l、x四种，所有的yaml文件都设置差不多，只有上面2和3的设置不同，作者团队很厉害，只需要修改这两个参数就可以调整模型的网络结构。

 ![img](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png)

#### parameters

负责模型的深度和宽度

```yaml
# parameters  
nc: 80  # number of classes  
depth_multiple: 0.33  # model depth multiple  
width_multiple: 0.50  # layer channel multiple  
```

`nc`：目标的类别数量。

`depth_multiple`：模型深度 控制模块的数量，当模块的数量number不为1时，模块的数量 = number * depth。

`width_multiple`：模型的宽度 控制卷积核的数量 ，卷积核的数量 = 数量 * width。

> depth_multiple 是用在backbone中的number≠1的情况下， 即在Bottleneck层使用，控制模型的深度，yolov5s中设置为0.33，假设yolov5l中有三个Bottleneck，那yolov5s中就只有一个Bottleneck。
> 因为一般number=1表示的是功能背景的层，比如说下采样Conv、Focus、SPP（空间金字塔池化）。
>
> width_multiple 主要是用于设置arguments，例如yolov5s设置为0.5，Focus就变成[32, 3]，Conv就变成[64, 3, 2]。
> 以此类推，卷积核的个数都变成了设置的一半。



#### Acnhor

```yaml
# anchors  
anchors:  
  - [10,13, 16,30, 33,23]  # P3/8  检测小目标  10，13是一组尺寸，一共三组  
  - [30,61, 62,45, 59,119]  # P4/16     
  - [116,90, 156,198, 373,326]  # P5/32  检测大目标  
```

yolov5已经在yaml预设好了输入图像为640*640分辨率对应的anchor尺寸，yolov5的anchor也是在大特征图上检测小目标，在小特征图上检测大目标。三个特征图，每个特征图上的格子有三种尺寸的anchor。

#### Backbone

```yaml
# YOLOv5 backbone  
backbone:  
  # from   第一列 输入来自哪一层  -1代表上一层，-2表示从上两层获得的输入， 4代表第4层     
  # number 第二列 卷积核的数量    最终数量需要乘上width  1表示只有一个，3表示有三个相同的模块
  # module 第三列 模块名称 包括：Conv Focus BottleneckCSP  SPP  
  # args   第四列 模块的参数   
  # [from, number, module, args]  
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2  
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4 卷积核的数量 = 128 * wedith = 128*0.5=64        
   [-1, 3, BottleneckCSP, [128]],       模块数量 = 3 * depth =3*0.33=1  
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8     
   [-1, 9, BottleneckCSP, [256]],       模块数量 = 9 * depth =9*0.33=3  
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16      
   [-1, 9, BottleneckCSP, [512]],  
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32      
   [-1, 1, SPP, [1024, [5, 9, 13]]],  
   [-1, 3, BottleneckCSP, [1024, False]],  # 9  
  ]  
```

* `Focus`：对特征图的切片操作，模块参数args分析： [[-1, 1, Focus, [64, 3]] 中的 [64, 3] 解析得到[3, 32, 3] ，输入为3（RGB），输出为64*0.5 = 32，3是卷积核 3*3

* `Conv`：这里的Conv由conv+Bn+Leaky_relu激活函数三者组成，模块参数args分析：[-1, 1, Conv, [128, 3, 2]]中的128 是卷积核数量，最终数量需要乘上width = 128 *0.5 = 64，3是卷积核 3*3，2是步长。
* `BottleneckCSP`：借鉴CSPNet网络结构，由三个卷积层和X个Res unint模块Concate组成，如果带有False参数就是没有使用Res unint模块，而是采用conv+Bn+Leaky_relu

* `SPP`：采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。

#### Head

```yaml
# YOLOv5 head  
head:  
  [[-1, 1, Conv, [512, 1, 1]],  
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],     上采样  
   [[-1, 6], 1, Concat, [1]],#cat backbone P4 [-1, 6]代表cat上一层和第6层  
   [-1, 3, BottleneckCSP, [512, False]],  # 13 第13层        
     
   [-1, 1, Conv, [256, 1, 1]],  
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],     上采样  
   [[-1, 4], 1, Concat, [1]],#cat backbone P3 [-1,4]代表cat上一层和第4层  
   [-1, 3, BottleneckCSP, [256, False]], # 17 (P3/8-small)    第17层  
  
   [-1, 1, Conv, [256, 3, 2]],  
   [[-1, 14], 1, Concat, [1]], #cat head P4 [-1,14]代表cat上一层和第14层  
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)  第20层  
  
   [-1, 1, Conv, [512, 3, 2]],                       
   [[-1, 10], 1, Concat, [1]], #cat head P5 [-1,10]代表cat上一层和第10层  
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)  第23层  
     
   [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)  
  ]#Detect [17, 20, 23] 代表输入的层数17和20和23  
```

Yolov5 Head 包括 Neck 和 Detector head 两部分，Neck部分使用了PANet的结构， Detector head使用和yolov3一样的head。其中，BottleneckCSP带有False参数说明没有使用Res unit结构而是采用了conv+Bn+Leaky_relu。



## 模型训练

为了训练我们的检测器，我们采取以下步骤：

- 安装 YOLOv5 依赖项
- 加载自定义 YOLOv5 数据集
- 定义 YOLOv5 模型配置和架构
- 训练自定义 YOLOv5 检测器
- 评估 YOLOv5 性能
- 可视化 YOLOv5 训练数据
- 在测试图像上运行 YOLOv5 推理
- 导出保存的 YOLOv5 权重以供将来推理



### Step01-安装 YOLOv5 依赖项

要从 YOLOv5 开始，我们首先克隆 YOLOv5 存储库并安装依赖项。

```python
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
```

![image-20210702201537419](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702201537419.png)

安装依赖环境和wandb包。

```python
!pip install -r requirements.txt
!pip install wandb
```

然后，我们可以看看我们从 Google Colab 免费提供给我们的训练环境。

```python
#查看torch版本和gpu
import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

[]:Setup complete. Using torch 1.9.0+cu102 (Tesla T4)
```

根据[Google官网](https://cloud.google.com/blog/products/ai-machine-learning/nvidia-tesla-t4-gpus-now-available-in-beta)显示，colab的T4应该只在部分地区开放，我这里用的是东京的ip来白嫖。

<img src="http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/T4_Regional_Availability.max-2200x2200.png" alt="gcp_nvidia_t4.png" style="zoom:50%;" />

之后我们测试是否安装成功

```python
#使用官方数据进行测试
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images
#需要注意路径，一般而言每运行一次都会新建一个exp
Image(filename='runs/detect/exp/zidane.jpg', width=600)
```





### Step02-加载自定义 YOLOv5 数据集

因为数据集不是很大，我这里最开始使用的google drive上传数据集的模式。

![image-20210702225219971](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702225219971.png)

目前colab已经默认挂载google网盘，放置在`/drive/`文件夹，因此我们直接解压就好

```python
!unzip -uq /content/drive/MyDrive/Yolov5_gw-detct_colab/Yolov5_gw_dataset.zip -d Yolov5_gw_dataset 
#现在是在yolov5文件夹内，所以直接解压到当前文件夹
```

![image-20210701203347221](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210701203347221.png)

### Step03-定义 YOLOv5 模型配置和架构



训练COCO数据集或者是VOC数据集，可以直接使用已经配置好的`coco.yaml` 和 `voc.yaml`文件。如果训练别的数据集，则需要模仿coco.yaml 文件写一个自己的`.yaml`文件保存在data文件夹下。我们创建数据集的配置文件`gw_detect.yaml`，指定了数据集的位置和分类信息。

```yaml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./Yolov5_gw_dataset/images/train/
val: ./Yolov5_gw_dataset/images/val/

# number of classes
nc: 2

# class names
names: [ 'gun', 'sword' ]
```

同样的我们将它上传到谷歌网盘，然后在colab使用。

### Step04-训练自定义 YOLOv5 检测器

我们设置完`gw_detect.yaml`后，就可以开始训练模型

为了开始训练，我们使用以下选项运行训练命令：

* `--img`：定义输入图像大小

* `--epochs`：迭代轮次，这里设置为500

* `--batch-size`：硬件不好的话，把16改为8或者4甚至更低（yolov5支持更小，这里白嫖的是colab的tesla T4，就直接16了）

* `--weights`：可以不加载预训练模型，这里改成无，当然也可以加载，这里使用yolov5s，具体模型选取看yolov5官方仓库的数据（也可以加载云盘中训练好的模型权重）

* `--data`：指定配置文件的位置

* `--cache`：缓存图像以加快训练速度

* `--upload_dataset`：wandb中的参数，在模型训练时将数据集作为DSViz表上传wandb，后续可以云端分享和读取
  * If the dataset is logged in you W&B dasboard, you get training progress artifacts which compares the predictions from each epoch with their ground truths（如果数据集记录在你的W&B数据板上，你会得到训练进度工件，它将每个历元的预测与它们的基本事实进行比较）

* `--save_preiod`：设置在将模型检查点记录为工件之前要等待的epoch数。如果未设置，则只记录最终训练的模型）

```python
!python train.py \
  --img 640 \
  --batch 16  \
  --epochs 500  \
  --data /content/drive/MyDrive/Yolov5_gw-detct_colab/gw_data.yaml \
  --weights yolov5s.pt  \
  --project Yolo-wandb-gw-detect \
  --upload_dataset \
  --save_period 50 \
  --cache 
```

接着就开始训练了，由于我们使用了wandb作为网络模型可视化，这里会弹出让我们填写api key的选项，点击链接进行填写就行。

![image-20210702200510369](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702200510369.png)

经过半个多小时，500个epoch跑完（比用K80快了十几分钟）

![image-20210702223125631](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702223125631.png)

* `last.pt`是最近一次迭代的模型
* `best.pt`是最好效果的模型

### 评估自定义 YOLOv5 检测器性能

现在我们已经完成了训练，我们可以通过查看验证指标来评估训练过程的执行情况。

最简单的方法是点击wandb的网页显示，直接查看wandb生成的结果图。

![image-20210702211656117](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702211656117.png)

在jupyter中，可以直接读取`/results.png`文件

```python
'''
如果要做图的话，可以参考官方colab
from utils.plots import plot_results 
plot_results(save_dir='runs/train/exp')  # plot all results*.txt files in 'runs/train/exp' 
# 'runs/train/exp'为运行文件目录，如果第二次运行，就是存放在exp2，要相应地进行修改
Image(filename='runs/train/exp/results.png', width=800)
'''
#这里wandb已经帮我作图了，直接读取就好
Image(filename='Yolo-wandb-gw-detect/exp/results.png', width=800)
```

![result](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/result.png)

### 可视化 YOLOv5 训练数据

在训练期间，YOLOv5 训练管道通过增强创建了一批训练数据。我们可以可视化训练数据的真实情况以及增强的训练数据

### 在测试图像上运行 YOLOv5 推理

现在我们使用我们训练好的模型并对测试图像进行推断。训练完成后，模型权重将保存在`weights/`。

对于推理，我们调用这些权重以及`--conf`指定的模型置信度（所需的置信度越高，预测越少）。

通过`source`来指定输入文件目录 。`source`可以接受图像目录、单个图像、视频文件以及设备的网络摄像头端口。

```python
!python detect.py 
	--weights Yolo-wandb-gw-detect/exp/weights/best.pt \
    --img 416 \
    --conf 0.4 \
    --source /content/drive/MyDrive/Yolov5_gw-detct_colab/gw_test_image/fig_sword_01.jpg
```

在这里我们使用网盘中的`Yolov5_gw-detct_colab/gw_test_image/fig_sword_01.jpg`进行测试

![image-20210702224615976](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702224615976.png)

可以看到在T4上只用了0.307s检测

由于已经运行过一次测试集（最开始测试yolov5安装是否成功的图片），所以这次检测测试输出在`exp2`文件夹。

```python
Image(filename='/content/yolov5/runs/detect/exp2/fig_sword_01.jpg',width=900)
```

看起来检测效果还行。

![image-20210702224458443](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702224458443.png)

把测试文件夹的其他图片也检测了

```python
#干脆全部检测一遍
!python detect.py 
  --weights Yolo-wandb-gw-detect/exp/weights/best.pt \
  --img 416 \
  --conf 0.4 \
  --source /content/drive/MyDrive/Yolov5_gw-detct_colab/gw_test_image/
```

七张图片共用时2.632s

![image-20210702224742448](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702224742448.png)

### 导出保存的 YOLOv5 权重以供将来推理

[colab](https://colab.research.google.com/drive/1CRC7kxYTIQQysowbrT2L6ctPen6ZQVeN?usp=sharing)



![image-20210702204142201](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702204142201.png)

## 参考资料

[**official YOLOv5 🚀 notebook**](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb#scrollTo=IEijrePND_2I)：This is the **official YOLOv5 🚀 notebook** authored by **Ultralytics**

[ Training a Custom Object Detection Model with YOLOv5](https://www.forecr.io/blogs/ai-algorithms/training-a-custom-object-detection-model-with-yolov5)：使用docker进行配置

[何在 NVIDIA Jetson 模块上使用 Docker 在 Pytorch 上运行 YoloV5 实时对象检测](https://www.forecr.io/blogs/ai-algorithms/how-to-run-yolov5-real-time-object-detection-on-pytorch-with-docker-on-nvidia-jetson-modules)

[使用google colab训练YOLOv5模型](https://xugaoxiang.com/2020/11/01/google-colab-yolov5/)

[YOLOv5学习总结（持续更新）](https://blog.csdn.net/weixin_38842821/article/details/108544609)

[史上最详细yolov5环境配置搭建+配置所需文件](https://blog.csdn.net/qq_44697805/article/details/107702939)

[大人时代变了](https://www.zhihu.com/question/334850317)

[YOLO5-王者荣耀 目标检测](https://www.bilibili.com/video/BV1g54y1a7jE?t=116)

[如何使用 Yolo、SORT 和 Opencv 跟踪足球运动员](https://towardsdatascience.com/how-to-track-football-players-using-yolo-sort-and-opencv-6c58f71120b8)

[YOLOV5训练代码train.py注释与解析](https://blog.csdn.net/Q1u1NG/article/details/107463417)

[wandb官网的介绍](https://wandb.ai/cayush/yolov5-dsviz-demo/reports/Object-Detection-with-YOLO-and-Weights-Biases--Vmlldzo0NTgzMjk)

[YOLOv5模型训练可视化](https://zhuanlan.zhihu.com/p/350955851)





## wandb





![image-20210702211439463](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702211439463.png)

![image-20210702200608586](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702200608586.png)

![image-20210702204237805](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702204237805.png)

![image-20210702204251512](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702204251512.png)

![image-20210702213805833](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702213805833.png)

https://wandb.ai/chancey/yolo-wandb-gw-detect/reports/---Vmlldzo4MjE0NDQ![image-20210702214043642](http://test-123-imagebed.oss-cn-beijing.aliyuncs.com/img/image-20210702214043642.png)



https://colab.research.google.com/drive/1Dwe00BGNNSOVaAXiXiW8GR9OJtpzF6Jc?usp=sharing

[**使用TensorRT对模型进行加速**](https://zhuanlan.zhihu.com/p/365191541)

[[tensorrtx](https://github.com/wang-xinyu/tensorrtx)/**yolov5**/](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)

https://wandb.ai/chancey/yolo-wandb-gw-detect/reports/---Vmlldzo4MjE0NzU

[Roboflow](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)

