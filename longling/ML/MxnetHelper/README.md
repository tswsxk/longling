## Overview
总共包含三个模块

### gallery
模型、网络层集合

* layer: 网络单元，如各类 attention，highway 等
* loss: 损失函数，如 pairwise-loss 等
* network: 网络层，如 TextCNN 等

### glue
模板模块，用以快速构建新的模型
模板文件在 glue/ModelName 下

### toolkit
工具包模块，包括一些专门适配于mxnet框架的辅助函数

包括
* 运行设备(cpu | gpu): ctx
* 优化器配置: optimizer_cfg
* 网络可视化: viz