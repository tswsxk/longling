## Introduction
Glue (Gluon Example) aims to generate a neural network model template of 
Mxnet-Gluon which can be quickly developed into a mature model. The source code
is [here](https://github.com/tswsxk/longling/tree/master/longling/ML/MxnetHelper/glue)

## Installation
It is automatically installed when you installing longling package. 
The tutorial of installing can be found 
[here](https://longling.readthedocs.io/zh/latest/tutorial.html#installation).

## Tutorial
Run the following commands to use glue:
```bash
# Create a full project including docs and requirements
glue --model_name ModelName
# Or, only create a network model template
glue --model_name ModelName --skip_top
# For more help
glue --help 
```
  