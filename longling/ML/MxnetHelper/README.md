总共包含四个模块

### helper
一些Mxnet常用的辅助函数，如判断混合编程模式下传入参数是Symbol还是NDArray的getF

[Full documentation](https://longling.readthedocs.io/zh/latest/submodule/ML/MxnetHelper/helper.html)

### gallery
模型、网络层集合

* layer: 网络单元，如各类 attention，highway 等
* loss: 损失函数，如 pairwise-loss 等
* network: 网络层，如 TextCNN 等

[Full documentation](https://longling.readthedocs.io/zh/latest/submodule/ML/MxnetHelper/toolkit.html)

### glue
模板模块，用以快速构建新的模型
模板文件在 [glue/ModelName](https://github.com/tswsxk/longling/tree/master/longling/ML/MxnetHelper/glue/ModelName) 下

[Full documentation](https://longling.readthedocs.io/zh/latest/submodule/ML/MxnetHelper/glue.html)

### toolkit
工具包模块，包括一些专门适配于mxnet框架的辅助函数

包括
* 运行设备(cpu | gpu): ctx
* 优化器配置: optimizer_cfg
* 网络可视化: viz

[Full documentation](https://longling.readthedocs.io/zh/latest/submodule/ML/MxnetHelper/gallery.html)
