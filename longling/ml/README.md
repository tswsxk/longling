## ML Framework

ML Framework is designed to help quickly construct a practical ML program where the user can focus on 
developing the algorithm despite of some other additional but important engineering components like `log` and `cli`.

Currently, two supported packages are provided for the popular DL framework: `mxnet` and `pytorch`. The overall scenery are almost the same, but the details may be a little different.

To be noticed that, ML Framework just provide a template, all components are allowed to be modified.

### Overview

The architecture produced by the ML Framework is look like:

```text
ModelName/
├── __init__.py
├── docs/
├── ModelName/
├── REAME.md
└── Some other components
```

And the core part is `ModelName` under the package `ModelName`, the architecture of it is:
```text
ModelName/
├── __init__.py
├── ModelName.py				<-- the main module
└── Module/
    ├── __init__.py
    ├── configuration.py		<-- define the configurable variables
    ├── etl.py					<-- define how the data will be loaded and preprocessed 
    ├── module.py				<-- the wrapper of the network, raraly need modification
    ├── run.py					<-- human testing script
    └── sym/					<-- define the network
        ├── __init__.py
        ├── fit_eval.py			<-- define how the network will be trained and evaluated
        ├── net.py				<-- network architecture
        └── viz.py				<-- (option) how to visualize the network
```

### Configuration

In configuration, some variables are predefined, such as the `data_dir`(where the data is stored) and `model_dir`(where the model file like parameters and running log should be stored).  The following rules is used to automatically construct the needed path, which can be modified as the user wants:

```python
model_name = "automatically be consistent with ModelName"

root = "./"
dataset = ""  # option
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # option
workspace = "" # option

root_data_dir = "$root/data/$dataset" if dataset else "$root/data"
data_dir = "$root_data_dir/data"
root_model_dir = "$root_data_dir/model/$model_name"
model_dir = "$root_model_dir/$workspace" if workspace else root_model_dir
cfg_path = "$model_dir/configuration.json"
```

The value of the variable containing `$` will be automatically evaluated during program running. Thus, it is easy to construct flexible variables via cli. For example, some one may want to have the `model_dir` contained `timestamp` information, then he or she can specify the `model_dir` in cli like:

```shell
--model_dir \$root_model_dir/\$workspace/\$timestamp
```

Annotation: `\` is a escape character in `shell`, which can have `\$variable` finally be got as the string `$variable` in the program, otherwise, the variable will be converted to the environment variable of `shell` like `$HOME`.

Also, some general variables which may frequently used in all algorithms are also predefined like `optimizer` and `batch_size`:

```python
# 训练参数设置
begin_epoch = 0
end_epoch = 100
batch_size = 32
save_epoch = 1

# 优化器设置
optimizer, optimizer_params = get_optimizer_cfg(name="base")
lr_params = {
    "learning_rate": optimizer_params["learning_rate"],
    "step": 100,
    "max_update_steps": get_update_steps(
        update_epoch=10,
        batches_per_epoch=1000,
    ),
}
```

