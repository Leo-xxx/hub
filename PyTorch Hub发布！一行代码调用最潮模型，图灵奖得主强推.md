## PyTorch Hub发布！一行代码调用最潮模型，图灵奖得主强推

关注前沿科技 [量子位](javascript:void(0);) *昨天*

##### 晓查 安妮 发自 凹非寺 量子位 出品 | 公众号 QbitAI

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwenXiciaibHcfYSh5O7rj2h2dicUwBYARiagxdptkaTicZTYpvyu89ds7DJkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

为了调用各种经典机器学习模型，今后你不必重复造轮子了。

刚刚，Facebook宣布推出**PyTorch Hub**，一个包含计算机视觉、自然语言处理领域的诸多经典模型的聚合中心，让你调用起来更方便。

有多方便？

图灵奖得主Yann LeCun强烈推荐，无论是ResNet、BERT、GPT、VGG、PGAN还是MobileNet等经典模型，**只需输入一行代码**，就能实现一键调用。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtDkzftEAKfzohadAYR3Ekrw8Tibx4eRkEygjphGKMwqE7OH7qtXVibrnRzxmJYONXDEeMrPNj6Ztictg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

厉不厉害！

Facebook官方博客表示，PyTorch Hub是一个简易API和工作流程，为复现研究提供了基本构建模块，包含**预训练模型库**。

并且，PyTorch Hub还支持Colab，能与论文代码结合网站Papers With Code集成，用于更广泛的研究。

发布首日已有**18个模型**“入驻”，获得英伟达官方力挺。而且Facebook还鼓励论文发布者把自己的模型发布到这里来，让PyTorch Hub越来越强大。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwmicjXer6iaiaeNyQSpr9v5gTpyuNHicFefGm1nfqsE3OnOVMdSpHiboWTlw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这个新工具一下子把不少程序员“圈了粉”。

短短几个小时，LeCun的推文就收获了上千条赞，网友好评如潮，花式称赞“Nice”“Great”“Wow”。

前Google Brain员工@mat kelcey调侃说，“Hub”这个词是机器学习模型项目的共享词么？TensorFlow Hub前脚到，PyTorch Hub就来了~

网友@lgor Brigadir跟评说，可能是从GitHub开始流行的。

所以，这个一问世就引发大批关注的PyTorch Hub，具体有哪些功能，该怎么用？来看看。

## 一行代码就导入

PyTorch Hub的使用简单到不能再简单，不需要下载模型，只用了一个torch.hub.load()就完成了对图像分类模型AlexNet的调用。

```
import torch
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
model.eval()
```

试看效果如何，可一键进入**Google Colab**运行。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwRqQsJZlOD7rXGVmZoMnHEBxzZRX1APS2ZspNeZnc1BesaBRdoYcFQw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

具体怎么用，Facebook分别给用户和发布者提供了指南。

### 对于用户

PyTorch Hub允许用户对已发布的模型执行以下操作：

> 1、查询可用的模型;
> 2、加载模型;
> 3、查询模型中可用的方法。

下面让我们来看看每个应用的实例。

#### 1、查询可用的模型

用户可以使用**torch.hub.list()**这个API列出repo中所有可用的入口点。比如你想知道PyTorch Hub中有哪些可用的计算机视觉模型：

```
>>> torch.hub.list('pytorch/vision')
>>>
['alexnet',
'deeplabv3_resnet101',
'densenet121',
...
'vgg16',
'vgg16_bn',
'vgg19',
 'vgg19_bn']
```

#### 2、加载模型

在上一步中能看到所有可用的计算机视觉模型，如果想调用其中的一个，也不必安装，只需一句话就能加载模型。

```
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
```

至于如何获得此模型的详细帮助信息，可以使用下面的API：

```
print(torch.hub.help('pytorch/vision', 'deeplabv3_resnet101'))
```

如果模型的发布者后续加入错误修复和性能改进，用户也可以非常简单地获取更新，确保自己用到的是最新版本：

```
model = torch.hub.load(..., force_reload=True)
```

对于另外一部分用户来说，稳定性更加重要，他们有时候需要调用特定分支的代码。例如pytorch_GAN_zoo的hub分支：

```
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=False)
```

#### 3、查看模型可用方法 

从PyTorch Hub加载模型后，你可以用dir(model)查看模型的所有可用方法。以bertForMaskedLM模型为例：

```
>>> dir(model)
>>>
['forward'
...
'to'
'state_dict',
]
```

如果你对forward方法感兴趣，使用help(model.forward) 了解运行运行该方法所需的参数。

```
>>> help(model.forward)
>>>
Help on method forward in module pytorch_pretrained_bert.modeling:
forward(input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None)
...
```

**PyTorch Hub中提供的模型也支持Colab。**

进入每个模型的介绍页面后，你不仅可以看到GitHub代码页的入口，甚至可以**一键进入Colab**运行模型Demo。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwEKj1pdOkSGccQXhh7rBX5srLWXylicdZHVH0Ay8sicxI4hvM3gWsyufA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 对于模型发布者

如果你希望把自己的模型发布到PyTorch Hub上供所有用户使用，可以去PyTorch Hub的GitHub页发送拉取请求。若你的模型符合高质量、易重复、最有利的要求，Facebook官方将会与你合作。

一旦拉取请求被接受，你的模型将很快出现在PyTorch Hub官方网页上，供所有用户浏览。

目前该网站上已经有18个提交的模型，英伟达率先提供支持，他们在PyTorch Hub已经发布了Tacotron2和WaveGlow两个TTS模型。

![img](https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtDkzftEAKfzohadAYR3Ekrw1vMnkQqcGFJFuhSoxbXY3kia4JwRcY6Y4aRicdTauzNa4Qiawwdzy3Hbg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

发布模型的方法也是比较简单的，开发者只需在自己的GitHub存储库中添加一个简单的**hubconf.py**文件，在其中枚举运行模型所需的依赖项列表即可。

比如，torchvision中的hubconf.py文件是这样的：

```
# Optional list of dependencies required by the package
dependencies = ['torch']

from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152,\
resnext50_32x4d, resnext101_32x8d
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from torchvision.models.googlenet import googlenet
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from torchvision.models.mobilenet import mobilenet_v2
```

Facebook官方向模型发布者提出了以下三点要求：

> 1、每个模型文件都可以独立运行和执行
> 2、不需要PyTorch以外的任何包
> 3、不需要单独的入口点，让模型在创建时可以无缝地开箱即用

Facebook还建议发布者最小化对包的依赖性，减少用户加载模型进行实验的阻力。

## 支持公开代码，从顶会做起

就在PyTorch Hub上线的同时，学术会议ICML 2019也开始在加州长滩举行。

和Facebook的理念相似，今年的ICML大会，首次鼓励研究人员提交代码以证明论文结果，增加了论文可重复性作为评审考察的因素，

也就是说，**开放代码更容易让你的论文通过评审**。

此前，挪威科技大学计算机科学家Odd Erik Gundersen调查后发现，过去几年在两个AI顶会上提出的400种算法中，只有**6％**的研究有公开代码。这就让长江后浪的直接调用非常困难了。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwDGAia0Vq01RGchvXN3uKgTp6z3dhc3sWEISEWoO50maIaN3gm9SFiazg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

ICML的政策也是顺应了不少研究人员的心声，这个政策施行的效果还不错。

据ICML 2019大会协同主席Kamalika Chaudhuri等人进行的统计显示，今年大约36％的提交论文和67％的已接受论文都共享了代码。

其中，来自学术界的研究人员的贡献热情比产业界高得多，学术界提交的作品中有90％的研究包含代码，而产业界只有27.4％。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtDkzftEAKfzohadAYR3EkrwuxffXrAK5GIqibr1mLuoSho8Y5EOanCfY2Y4Ow1cJ7VK8TuVEI0X4qA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

与之相比，NeurIPS 2018的已接收论文中，只有不到一半的论文附上了代码。

总之，对于AI领域的长远发展来说，这是个大好现象~

## 传送门

官方介绍博客：
https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/

测试版PyTorch Hub：
https://pytorch.org/hub

PyTorch Hub的GitHub主页：
https://github.com/pytorch/hub

— **完** —

**AI内参|关注行业发展**

![img](https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtDcZyEBVM81oW4VRoNAibJWwhmzcUTiaV6NZeRQ3JLupxwsLmpc39ONqZ2KtMLPcAto6ly3qapjsrLg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**AI社群|与优秀的人交流**

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



**量子位** QbitAI · 头条号签约作者





վ'ᴗ' ի 追踪AI技术和产品新动态



喜欢就点「好看」吧 !











微信扫一扫
关注该公众号