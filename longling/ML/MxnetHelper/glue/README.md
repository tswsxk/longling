## Introduction
Glue (Gluon Example) aims to generate a neural network model template of 
Mxnet-Gluon which can be quickly developed into a mature model. The source code
is [here](https://github.com/tswsxk/longling/tree/master/longling/ML/MxnetHelper/glue)

## Installation
It is automatically installed when you installing longling package. 
The tutorial of installing can be found 
[here](https://longling.readthedocs.io/zh/latest/tutorial.html#installation).

## Tutorial
With glue, it is possible to quickly construct a model.

### Generate template files 
Run the following commands to use glue:
```bash
# Create a full project including docs and requirements
glue --model_name ModelName
# Or, only create a network model template
glue --model_name ModelName --skip_top
```
The template files will be generate in current directory. To change 
the position of files, use `--directory` option to specify the location:
```bash
glue --model_name ModelName --directory LOCATION
```
For more help, run `glue --help`

### Guidance to modify the template
#### Overview
Usually, the project template will consist of doc files and model files. Assume
the project name by default is ModelName,then the directory of model files will
have the same name, the directory tree is like:
```
ModelName(Project)
    ----docs
    ----ModelName(Model) 
```
#### Construct the model

##### Build and test 



  