# coding: utf-8
# create by tongshiwei on 2018/8/5

import argparse
import inspect
import os

import yaml

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.stream import wf_open
from longling.lib.candylib import as_list

from mxnet import Context, cpu, gpu


class MXCtx(object):
    @staticmethod
    def load(data):
        ctx_vars = []
        for device_type, device_ids in data.items():
            if isinstance(device_ids, int):
                device_ids = as_list(device_ids)
            elif isinstance(device_ids, str):
                device_ids = map(int, device_ids.split(','))
            for device_id in device_ids:
                ctx_vars.append(eval(device_type)(device_id))
        return ctx_vars

    @staticmethod
    def dump(data):
        ctx_vars = {}
        for ctx in as_list(data):
            assert isinstance(ctx, Context)
            if ctx.device_type not in ctx_vars:
                ctx_vars[ctx.device_type] = []
            ctx_vars[ctx.device_type].append(ctx.device_id)
        for device_type, device_ids in ctx_vars.items():
            if len(device_ids) > 1:
                ctx_vars[device_type] = ",".join(list(map(str, device_ids)))
            else:
                ctx_vars[device_type] = device_ids[0]
        return ctx_vars


class Parameters(object):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    model_name = "module_name"
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)

    data_dir = os.path.abspath(os.path.join(root, "data"))
    model_dir = os.path.abspath(os.path.join(data_dir, "module_name"))

    optimizer = 'sgd'
    optimizer_params = {
        'learning_rate': .01,
        'wd': 0.5,
        'momentum': 0.01,
        'clip_gradient': 5,
    }

    begin_epoch = 0
    end_epoch = 100
    batch_size = 128

    ctx = cpu()

    def __init__(self, params_yaml=None, **kwargs):
        params = self.class_var
        if params_yaml:
            params.update(self.load(params_yaml=params_yaml))
        params.update(**kwargs)
        for param, value in params.items():
            setattr(self, "%s" % param, value)

    def items(self):
        return {k: v for k, v in vars(self).items() if k not in {'logger'}}

    @staticmethod
    def load(params_yaml):
        f = open(params_yaml)
        params = yaml.load(f)
        if 'ctx' in params:
            params['ctx'] = MXCtx.load(params['ctx'])
        f.close()
        return params

    def dump(self, param_yaml, override=False):
        if os.path.isfile(param_yaml) and not override:
            self.logger.warning("file %s existed, dump aborted" % param_yaml)
            return
        self.logger.info("writing parameters to %s" % param_yaml)
        with wf_open(param_yaml) as wf:
            store_vars = {k: v for k, v in vars(self).items() if k not in {'logger'}}
            if 'ctx' in store_vars:
                store_vars['ctx'] = MXCtx.dump(store_vars['ctx'])
            dump_data = yaml.dump(store_vars, default_flow_style=False)
            print(dump_data, file=wf)
            self.logger.info(dump_data)

    @property
    def class_var(self):
        variables = {k: v for k, v in vars(type(self)).items() if
                     not inspect.isroutine(v) and k not in {'__doc__', '__module__', '__dict__', '__weakref__',
                                                            'class_var'}}
        return variables


class ParameterParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ParameterParser, self).__init__(*args, **kwargs)
        self.add_argument('--root_prefix', dest='root_prefix', default='', help='set root prefix')
        params = {k: v for k, v in vars(Parameters).items() if
                  not inspect.isroutine(v) and k not in {'__doc__', '__module__', '__dict__', '__weakref__',
                                                         'class_var'}}
        for param, value in params.items():
            if param == 'logger':
                continue
            self.add_argument('--%s' % param, help='set %s, default is %s' % (param, value), default=value)
        self.add_argument('--kwargs', required=False, help=r"add extra argument here, use format: <key>=<value>")

    @staticmethod
    def parse(arguments):
        arguments = vars(arguments)
        args_dict = dict()
        for k, v in arguments.items():
            if k in {'root_prefix'}:
                continue
            args_dict[k] = v
        if arguments['root_prefix']:
            args_dict['root'] = os.path.abspath(os.path.join(arguments['root_prefix'], arguments['root']))

        return args_dict


if __name__ == '__main__':
    default_yaml_file = os.path.join(Parameters.model_dir, "parameters.yaml")

    # 命令行参数配置
    parser = ParameterParser()
    kwargs = parser.parse_args()
    kwargs = parser.parse(kwargs)

    data_dir = os.path.join(Parameters.root, "data")
    parameters = Parameters(
        **kwargs
    )
    parameters.dump(default_yaml_file, override=True)
