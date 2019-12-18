## Introduction
Glue (Gluon Example) aims to generate a neural network model template of 
Mxnet-Gluon which can be quickly developed into a mature model. The source code
is [here](https://github.com/tswsxk/longling/tree/master/longling/ML/MxnetHelper/glue)

## Installation
It is automatically installed when you installing longling package. 
The tutorial of installing can be found 
[here](https://longling.readthedocs.io/zh/latest/tutorial.html#installation).

## Tutorial
With glue, it is possible to quickly construct a model. A demo case can be 
referred in [here](). And the model can be divided into several different 
functionalities:

* ETL(extract-transform-load): generate the data stream for model;
* Symbol()

Also, we call those variables like working directory, path to data, 
hyper parameters  

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

And in ModelName(Model), there are one template file named ModelName.py 
and a directory containing four sub-template files.

The directory tree is like:
```text
ModelName/
├── __init__.py
├── ModelName.py
└── Module/
    ├── __init__.py
    ├── configuration.py
    ├── etl.py
    ├── module.py
    ├── run.py
    └── sym/
        ├── __init__.py
        ├── fit_eval.py
        ├── net.py
        └── viz.py
```
* The [configuration.py]() defines the all parameters should be configured, 
like where to store the model parameters and configuration parameters, 
the hyper-parameters of the neural network.
* The [etl.py]() defines the process of extract-transform-load, which is 
the definition of data processing.
* The [module.py]() serves as a high-level wrapper for sym.py, which provides
the well-written interfaces, like model persistence, batch loop, epoch loop and 
data pre-process on distributed computation.
* The [sym.py]() is the minimal model can be directly used to train, evaluate, 
also supports visualization. But some higher-level operations are not supported
for simplification and modulation, which are defined in module.py.

#### Data Stream
* extract: extract the data from data src
```python
def extract(data_src):
    # load data from file, the data format is looked like:
    # feature, label
    features = []
    labels = []
    with open(data_src) as f:
        for line in f:
            feature, label = line.split()
            features.append(feature)
            labels.append(label)
    return features, labels
``` 
* transform:
Convert the extracted into batch data. 
The pre-process like bucketing can be defined here.
```python
from mxnet import gluon
def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size

    return gluon.data.DataLoader(gluon.data.ArrayDataset(raw_data), batch_size)
```
* etl: combine the extract and transform together.

#### Model Construction
Usually, there are three level components need to be configured:
1. bottom: the network symbol and how to fit and eval it;
2. middle: the higher level to define the batch and epoch, also 
the initialization and persistence of model parameters.
3. top: the api of model

##### Bottom
###### configuration
Find the configuration.py and define the configuration variables that you need,
for example:

* begin_epoch
* end_epoch
* batch_size

Also, the paths can be configured:

```python
import longling.ML.MxnetHelper.glue.parser as parser
from longling.ML.MxnetHelper.glue.parser import var2exp
import pathlib
import datetime

 # 目录配置
class Configuration(parser.Configuration):
    model_name = str(pathlib.Path(__file__).parents[1].name)
    
    root = pathlib.Path(__file__).parents[2]
    dataset = ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    workspace = ""
    
    root_data_dir = "$root/data/$dataset" if dataset else "$root/data"
    data_dir = "$root_data_dir/data"
    root_model_dir = "$root_data_dir/model/$model_name"
    model_dir = "$root_model_dir/$workspace" if workspace else root_model_dir
    cfg_path = "$model_dir/configuration.json"
    
    def __init__(self, params_json=None, **kwargs):
        # set dataset
        if kwargs.get("dataset"):
            kwargs["root_data_dir"] = "$root/data/$dataset"
        # set workspace
        if kwargs.get("workspace"):
            kwargs["model_dir"] = "$root_model_dir/$workspace"

        # rebuild relevant directory or file path according to the kwargs
        _dirs = [
            "workspace", "root_data_dir", "data_dir", "root_model_dir",
            "model_dir"
        ]
        for _dir in _dirs:
            exp = var2exp(
                kwargs.get(_dir, getattr(self, _dir)),
                env_wrap=lambda x: "self.%s" % x
            )
            setattr(self, _dir, eval(exp))
```
How the variable paths work can be referred in [here]() 

Refer to the [prototype](https://github.com/tswsxk/longling/blob/master/longling/ML/MxnetHelper/glue/ModelName/ModelName/Module/configuration.py) for illustration.
Refer to [full documents about Configuration]() for details.
 
###### build the network symbol and test it
The network symbol file is [sym.py]()

The following variables and functions should be rewritten (marked as **_todo_**):


two ways can be used to check whether the network works well:

1. Visualization:
* functions name: net_viz
*
* 
2. Numerical:
* 
*
*





  