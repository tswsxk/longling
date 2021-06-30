# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = [
    "template_copy",
    "gitignore",
    "pytest", "coverage",
    "pysetup", "sphinx_conf", "makefile",
    "readthedocs", "travis", "nni",
    "dockerfile", "gitlab_ci", "chart"
]

import functools
import os
from collections import OrderedDict
from shutil import copyfile as _copyfile, rmtree, copytree as _copytree

from longling import wf_open, PATH_TYPE
from longling.Architecture.utils import binary_legal_input
from longling.lib.path import abs_current_dir, path_append
from longling.lib.regex import default_variable_replace as dvr
from longling.lib.utilog import config_logging, LogLevel
from longling.lib.yaml_helper import FoldedString, ordered_yaml_load, dump_folded_yaml

from . import config

logger = config_logging(logger="arch", console_log_level=LogLevel.INFO)

META = path_append(abs_current_dir(__file__), "meta_docs")
default_variable_replace = functools.partial(dvr, quotation="\'")


def copytree(src, dst, **kwargs):
    """
    Recursively copy a directory tree.

    Change OVERRIDE mode to specify the operation when dst existed.

    Examples
    --------
    .. code-block :: python

        # to change the OVERRIDE operation
        from longling.Architecture import config
        config.OVERRIDE = False  # skip existed dst
        config.OVERRIDE = None  # use console input to determine
        config.OVERRIDE = False  # override existed dst
    """
    if not override_check(dst):
        return
    elif os.path.exists(dst):
        rmtree(dst)
    _copytree(src, dst, **kwargs)


def copyfile(src, dst):
    """
    Copy data from src to dst.

    Change OVERRIDE mode to specify the operation when dst existed.

    Examples
    --------
    .. code-block :: python

        # to change the OVERRIDE operation
        from longling.Architecture import config
        config.OVERRIDE = False  # skip existed dst
        config.OVERRIDE = None  # use console input to determine
        config.OVERRIDE = False  # override existed dst
    """
    if override_check(dst):
        _copyfile(src, dst)


def override_check(path) -> bool:
    """
    Whether to override the specified file or directory

    not override:
     * config.OVERRIDE is False
     * config.OVERRIDE is None and user confirm not to override by console input

    Returns
    -------
    override: bool
        True to override while False not
    """
    if os.path.exists(path):
        if config.OVERRIDE is False:
            logger.error("%s exists, skipped" % path)
            return False
        elif config.OVERRIDE is None and not binary_legal_input("%s exists, override?" % path, _default="n"):
            logger.info("skip %s" % path)
            return False
        else:
            logger.info("%s exists, overrided" % path)
    return True


def template_copy(src: PATH_TYPE, tar: PATH_TYPE,
                  default_value: (str, dict, None) = "", quotation="\'", key_lower=True,
                  **variables):
    """
    Generate the tar file based on the template file where the variables will be replaced.
    Usually, the variable is specified like `$PROJECT` in the template file.


    Parameters
    ----------
    src: template file
    tar: target location
    default_value: the default value
    quotation: the quotation to wrap the variable value
    variables: the real variable values which are used to replace the variable in template file
    """

    if not override_check(tar):
        return

    with open(src) as f, wf_open(tar) as wf:
        for line in f:
            print(
                default_variable_replace(
                    line, default_value=default_value, quotation=quotation, key_lower=key_lower,
                    **variables
                ),
                end='', file=wf
            )


def gitignore(atype: str = "", tar_dir: PATH_TYPE = "./"):
    """
    cli alias: ``arch gitignore``


    Parameters
    ----------
    atype: the gitignore type, currently support `docs` and `python`
    tar_dir: target directory

    """
    src = path_append(META, "gitignore", "%s.gitignore" % atype)
    tar = path_append(tar_dir, ".gitignore")
    logger.info("gitignore: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def pytest(tar_dir: PATH_TYPE = "./"):
    """
    cli alias: ``arch pytest``

    Parameters
    ----------
    tar_dir

    Returns
    -------

    """
    src = path_append(META, "pytest.ini")
    tar = path_append(tar_dir, "pytest.ini")
    logger.info("pytest: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def coverage(tar_dir: PATH_TYPE = "./", **variables):
    """
    cli alias: ``arch coverage``

    Parameters
    ----------
    tar_dir:
    variables:
        These variables should be provided:

        * project

    """
    src = path_append(META, "setup.cfg.template")
    tar = path_append(tar_dir, "setup.cfg")
    logger.info("coverage: template %s -> %s" % (src, tar))
    template_copy(src, tar, quotation="", **variables)


def pysetup(tar_dir="./", **variables):
    """
    cli alias: ``arch pysetup``

    Parameters
    ----------
    tar_dir
    variables

    Returns
    -------

    """
    src = path_append(META, "setup.py.template")
    tar = path_append(tar_dir, "setup.py")
    logger.info("pysetup: template %s -> %s" % (src, tar))
    template_copy(src, tar, **variables)


def sphinx_conf(tar_dir="./", **variables):
    """
    cli alias: ``arch sphinx_conf``

    Parameters
    ----------
    tar_dir
    variables

    Returns
    -------

    """
    if variables["docs_style"] == "mxnet":
        src = path_append(META, "docs/mxnet/conf.py.template")
        tar = path_append(tar_dir, "conf.py")
        logger.info("sphinx_conf: template %s -> %s" % (src, tar))
        template_copy(src, tar, **variables)

        src = path_append(META, "docs/mxnet/.math.json")
        tar = path_append(tar_dir, ".math.json")
        logger.info("sphinx_conf: copy %s -> %s" % (src, tar))
        copyfile(src, tar)

        src = path_append(META, "docs/mxnet/requirements.txt")
        tar = path_append(tar_dir, "requirements.txt")
        logger.info("sphinx_conf: copy %s -> %s" % (src, tar))
        copyfile(src, tar)

        logger.warning(
            "\n%s\nmodify setup.py according to the components in %s\n%s\n" % ('*' * 60, tar, '*' * 60)
        )
    else:
        src = path_append(META, "docs/sphinx/requirements.txt")
        tar = path_append(tar_dir, "requirements.txt")
        logger.info("sphinx_conf: copy %s -> %s" % (src, tar))
        copyfile(src, tar)
        logger.warning(
            "\n%s\nmanually run 'sphinx-quickstart' in %s to create necessary components\n%s" % (
                '*' * 60, tar_dir, '*' * 60)
        )


def readthedocs(tar_dir="./"):
    """
    cli alias: ``arch readthedocs``

    Parameters
    ----------
    tar_dir

    Returns
    -------

    """
    src = path_append(META, "docs/.readthedocs.yml")
    tar = path_append(tar_dir, ".readthedocs.yml")
    logger.info("readthedocs: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def dockerfile(atype, tar_dir="./", **variables):
    """
    cli alias: ``arch dockerfile``

    Parameters
    ----------
    atype
    tar_dir
    variables

    Returns
    -------

    """
    src = path_append(META, "Dockerfile/%s.docker" % atype)
    tar = path_append(tar_dir, "Dockerfile")
    logger.info("Dockerfile:  %s -> %s" % (src, tar))
    template_copy(src, tar, quotation="", **variables)


def travis(tar_dir: PATH_TYPE = "./"):
    """
    cli alias: ``arch travis``

    Parameters
    ----------
    tar_dir

    Returns
    -------

    """
    src = path_append(META, ".travis.yml")
    tar = path_append(tar_dir, ".travis.yml")
    logger.info("travis: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def chart(tar_dir: PATH_TYPE = "./"):
    """
    cli alias: ``arch chart``

    Parameters
    ----------
    tar_dir: target directory

    """

    src_dir = path_append(META, "chart")
    tar_dir = path_append(tar_dir, "chart/")
    logger.info("chart: copy %s -> %s" % (src_dir, tar_dir))
    copytree(src_dir, tar_dir)


def helm_service(host="${KUBE_NAMESPACE}", image_repo="${CI_REGISTRY_IMAGE}", image_port=None, private=True,
                 name="$KUBE_NAMESPACE",
                 image_tag="latest", path_to_api=""):
    """

    Parameters
    ----------
    host
    image_repo
    image_port
    private
    name
    image_tag
    path_to_api

    Returns
    -------

    """
    src = path_append(META, "gitlab-ci/helm_install.gitlab-ci.yml.template")

    variables = {
        "NAME": name,
        "IMAGE_TAG": image_tag,
        "IMAGE_REPO": image_repo,
        "HOST": host,
        "PATH_TO_API": path_to_api,
    }

    with open(src, encoding="utf-8") as f:
        helm_install_commands = "".join(f.readlines()).strip() + "\n"
        helm_install_commands = default_variable_replace(helm_install_commands,
                                                         key_lower=False, quotation="", **variables)

    if image_port:
        helm_install_commands += "--set \"image.port=%s\"" % image_port + "\n"

    if private:
        helm_install_commands += "--set \"dockercfg=$(cat /root/.docker/config.json | base64 | tr -d '\\n')\"" + "\n"

    if not path_to_api:
        helm_install_commands += "--set \"ingress.annotations=null\"" + "\n"

    return FoldedString(helm_install_commands)


def _gitlab_ci(commands: dict, stage, image_name, private=True, on_stop=None, manual=False, only_master=False,
               deployment=True, registry_suffix="", version_in_path=True, **kwargs):
    name = "$KUBE_NAMESPACE"

    _commands = commands.get(stage, OrderedDict())
    _commands["image"] = image_name
    script = _commands.get("script", [])
    if script is None:  # pragma: no cover
        script = []
        _commands["script"] = script

    image_tag = "${API_VERSION}"
    ci_registry_image = "${CI_REGISTRY_IMAGE}"
    environment_name = stage

    export_version = r"export API_VERSION=$(cat chart/Chart.yaml | grep apiVersion | cut -d\  -f2)"

    if not only_master:
        image_tag = "${CI_COMMIT_REF_NAME}"
        environment_name += "/${CI_COMMIT_REF_NAME}"
    else:
        name += "-$API_VERSION"

    uninstall_heml = "helm uninstall %s || true" % name
    if deployment:
        docker_registry_image = "%s:%s" % ("${CI_REGISTRY_IMAGE}", image_tag)
        script[0] = dvr(script[0], key_lower=False,
                        **{"DOCKER_REGISTRY_IMAGE": docker_registry_image})
        if registry_suffix:
            ci_registry_image += registry_suffix
            script.insert(0, "export CI_REGISTRY_IMAGE=%s" % ci_registry_image)
        script.insert(0, export_version)

        script.append("helm dep build chart")
        script.append(uninstall_heml)
        if version_in_path:
            path_to_api = "${API_VERSION}(/|$)(.*)"
        else:
            path_to_api = ""
            kwargs.update({"host": "${KUBE_NAMESPACE}-%s" % "${API_VERSION}"})

        script.append(helm_service(name=name, private=private, path_to_api=path_to_api, image_tag=image_tag, **kwargs))

        environment = OrderedDict({"name": environment_name, "url": "https://$KUBE_NAMESPACE.env.bdaa.pro"})

        if on_stop is not None:
            environment["on_stop"] = "stop_%s" % stage
            if not manual:
                environment["auto_stop_in"] = "1 week"
            stop_commands = OrderedDict({
                "stage": "%s" % stage,
                "image": image_name,
                "script": uninstall_heml,
                "when": "manual",
                "variables": {"GIT_STRATEGY": "none"},
                "environment": OrderedDict({
                    "name": environment_name,
                    "action": "stop",
                }),
            })
        else:
            stop_commands = None

        _commands["environment"] = environment
        if only_master:
            _commands["only"] = OrderedDict({"refs": ["master"], "kubernetes": "active"})

        if manual:
            _commands["when"] = "manual"

        commands[stage] = _commands

        if stop_commands is not None:
            commands["stop_review"] = stop_commands


def gitlab_ci(private, stages: dict, atype: str = "", tar_dir: PATH_TYPE = "./", version_in_path=True):
    """
    cli alias: ``arch gitlab_ci``

    Parameters
    ----------
    private
    stages
    atype
    tar_dir
    version_in_path

    Returns
    -------

    """
    base_src = path_append(META, "gitlab-ci", ".gitlab-ci.yml")
    src = path_append(META, "gitlab-ci", "%s.gitlab-ci.yml" % atype)
    tar = path_append(tar_dir, ".gitlab-ci.yml")

    config_template = OrderedDict()

    with open(base_src) as f:
        config_template.update(ordered_yaml_load(f))

    with open(src) as f:
        config_template.update(ordered_yaml_load(f))

    logger.info("generate %s" % tar)

    with wf_open(tar) as wf:
        for _c in ["variables", "cache"]:
            if _c in config_template:
                print(dump_folded_yaml({_c: config_template[_c]}), file=wf)

        print(dump_folded_yaml({"stages": [stage for stage in stages.keys() if stage in config_template]}), file=wf)

        for stage, params in stages.items():
            if stage == "docs":
                params["registry_suffix"] = "/docs"
            elif stage in {"test", "build"}:
                params["deployment"] = False

            if stage not in config_template:
                logger.warning("%s is not listed in %s, skipped" % (stage, src))
                continue
            commands = {stage: config_template[stage]}
            _gitlab_ci(commands, stage, private=private, version_in_path=version_in_path, **params)
            print(dump_folded_yaml(commands), file=wf)

    if private:
        print("*" * 30)
        print("私有项目注意")
        print(
            "在项目的settings->repository->Deploy Tokens 添加一个name为gitlab-deploy-token、"
            "其他两项留空的具有read_registry权限的token"
        )
        print("*" * 30)


def makefile(tar_dir="./", **variables):
    """
    cli alias: ``arch makefile``

    Parameters
    ----------
    tar_dir
    variables

    Returns
    -------

    """
    src = path_append(META, "Makefile.template")
    tar = path_append(tar_dir, "Makefile")
    logger.info("makefile: template %s -> %s" % (src, tar))
    template_copy(src, tar, default_value=None, quotation='', **variables)


def nni(tar_dir="./"):
    """
    cli alias: ``arch nni`` and ``install nni``

    Parameters
    ----------
    tar_dir

    Returns
    -------

    """
    src_dir = path_append(META, "nni")
    for file in ["_config.yml", "_search_space.json"]:
        copyfile(path_append(src_dir, file), path_append(tar_dir, file))
