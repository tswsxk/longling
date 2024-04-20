# coding: utf-8
# 2020/4/14 @ tongshiwei

from longling.infrastructure.cli.main import cli
from longling.utils.testing import simulate_stdin


def test_python_cli(tmp_path_factory):
    """
    without docs:
        Project Name (default is test) <
        Project Type (python/docs/web) < python
        Install travis? (y/n, default is y) <
        Install docs? (y/n, default is y) < n
        To deploy as a service (y/n, default is y) < n
        Install Dockerfile? (y/n, default is y) <
        Choose a service type (flask/cli) < cli
        Choose a image (default is python:3.6) <
        Specify the main entry (e.g., the path to main.py) < main.py

    with docs:
        Project Name (default is test) <
        Project Type (python/docs/web) < python
        Install travis? (y/n, default is y) < n
        Install docs? (y/n, default is y) <
        Docs Style (sphinx/mxnet, default is sphinx) <
        Make 'docs/' directory? (y/n, default is y) <
        Install .readthedocs.yml? (y/n, default is y) <
        Install Dockerfile for documents? (y/n, default is y) <
        Choose a image (default is nginx) <
        Specify the html directory (default is _build/html) <
        To deploy as a service (y/n, default is y) <
        Choose a service type (cli/flask) < cli
        Choose a image (default is python:3.6) <
        Specify the main entry (e.g., the path to main.py) < main.py
        Image Port < 80
        Is private project? (y/n, default is y) < n
        Install .gitlab-ci.yml? (y/n, default is y) <
        Need [build] Stage? (y/n, default is n) <
        Need [test] Stage? (y/n, default is y) < n
        Need [review] Stage? (y/n, default is y) < n
        Need [docs] Stage? (y/n, default is y) <
        Choose a image < nginx
        Stage Image Port (default is 80) <
        Only triggered in master branch? (y/n, default is y) <
        Triggered manually? (y/n, default is y) <
        Need [production] Stage? (y/n, default is y) < n


    """
    project = tmp_path_factory.mktemp("python")
    inputs = ["arch", "python"] + ["", "n", "n", ""] + ["cli"] + ["", "main.py"]
    with simulate_stdin(*inputs):
        cli(tar_dir=project)

    inputs = ["python_docs", "python"] + ["n"] + [""] * 8 + ["cli"] + ["", "main.py"] + [
        "80"] + ["n", "", "", "n", "n"] + ["", "nginx", "", "", ""] + ["n"]
    with simulate_stdin(*inputs):
        cli(skip_top=False, tar_dir=project)


def test_docs_cli(tmp_path_factory):
    """
    Project Name (default is test) <
    Project Type (python/docs/web) < docs
    Install travis? (y/n, default is y) <
    Install docs? (y/n, default is y) <
    Docs Style (mxnet/sphinx, default is mxnet)? <
    Make 'docs/' directory? (y/n, default is y) <
    Docs Title (default is test) <
    Author < Shiwei Tong
    Copyright (default is 2020, Shiwei Tong) <
    Install .readthedocs.yml? (y/n, default is y) <
    Install Dockerfile for documents? (y/n, default is y) <
    Choose a image (default is nginx) <
    Specify the html directory (default is _build/html) <
    """
    project = tmp_path_factory.mktemp("docs")
    inputs = ["arch", "docs"] + [""] * 5 + ["Sherlock"] + [""] * 5
    with simulate_stdin(*inputs):
        cli(tar_dir=project)


def test_web_cli(tmp_path_factory):
    """
    Project Name (default is test) <
    Project Type (python/docs/web) < web
    Install travis? (y/n, default is y) <
    Install docs? (y/n, default is y) <
    Docs Style (mxnet/sphinx, default is sphinx)? <
    Make 'docs/' directory? (y/n, default is y) <
    Install .readthedocs.yml? (y/n, default is y) <
    Install Dockerfile for documents? (y/n, default is y) <
    Choose a image (default is nginx) <
    Specify the html directory (default is _build/html) <
    To deploy as a service (y/n, default is y) <
    Choose a service type (vue/nginx) < vue
    Specify the html directory (default is build/html) <
    Image Port (default is None) <
    Is private project? (y/n, default is y) <
    Install .gitlab-ci.yml? (y/n, default is y) <
    Need [build] Stage? (y/n, default is y) <
    Choose a image < hello
    Need [test] Stage? (y/n, default is y) <
    Choose a image (default is hello) <
    Need [review] Stage? (y/n, default is y) <
    Choose a image < hello
    Add Corresponding Stop Stage? (y/n, default is y) <
    Only triggered in master branch? (y/n, default is n) <
    Triggered manually? (y/n, default is n) <
    Need [docs] Stage? (y/n, default is y) <
    Choose a image < world
    Stage Image Port (default is 80) <
    Only triggered in master branch? (y/n, default is y) <
    Triggered manually? (y/n, default is y) <
    Need [production] Stage? (y/n, default is y) <
    Choose a image (default is hello) <
    Only triggered in master branch? (y/n, default is y) <
    Triggered manually? (y/n, default is y) <
    """
    project = tmp_path_factory.mktemp("web")
    inputs = ["arch", "web"] + [""] * 9 + ["vue"] + [""] * 4 + [""] * 5 + [
        "hello"] + [""] * 3 + ["hello"] + [""] * 4 + ["world"] + [""] * 7
    with simulate_stdin(*inputs):
        cli(tar_dir=project)


def test_deploy_cli(tmp_path_factory):
    """
    Web nginx:
        Project Name (default is test) <
        Project Type (python/docs/web) < web
        Install travis? (y/n, default is y) < n
        Install docs? (y/n, default is y) < n
        To deploy as a service (y/n, default is y) <
        Choose a service type (vue/nginx) < nginx
        Choose a image (default is nginx) <
        Specify the html directory (default is build/html) <
        Image Port (default is None) <
        Is private project? (y/n, default is y) <
        Install .gitlab-ci.yml? (y/n, default is y) <
        Need [build] Stage? (y/n, default is y) < n
        Need [test] Stage? (y/n, default is y) < n
        Need [review] Stage? (y/n, default is y) < n
        Need [production] Stage? (y/n, default is y) < n

    Web nginx with docs:
        Project Name (default is test) <
        Project Type (python/docs/web) < web
        Install travis? (y/n, default is y) < n
        Install docs? (y/n, default is y) <
        Docs Style (mxnet/sphinx, default is sphinx) <
        Make 'docs/' directory? (y/n, default is y) <
        Install .readthedocs.yml? (y/n, default is y) <
        Install Dockerfile for documents? (y/n, default is y) <
        Choose a image (default is nginx) <
        Specify the html directory (default is _build/html) <
        To deploy as a service (y/n, default is y) <
        Choose a service type (vue/nginx) < nginx
        Choose a image (default is nginx) <
        Specify the html directory (default is build/html) <
        Image Port < 80
        Is private project? (y/n, default is y) < n
        Install .gitlab-ci.yml? (y/n, default is y) <
        Need [build] Stage? (y/n, default is y) <
        Choose a image < nginx
        Need [test] Stage? (y/n, default is y) < n
        Need [review] Stage? (y/n, default is y) < n
        Need [docs] Stage? (y/n, default is y) <
        Choose a image < nginx
        Stage Image Port (default is 80) <
        Only triggered in master branch? (y/n, default is y) < n
        Triggered manually? (y/n, default is y) < n
        Need [production] Stage? (y/n, default is y) < n

    Python 1:
        Project Name (default is test) <
        Project Type (python/docs/web) < python
        Install travis? (y/n, default is y) < n
        Install docs? (y/n, default is y) < n
        To deploy as a service (y/n, default is y) <
        Choose a service type (flask/cli) < flask
        Choose a image (default is python:3.6) <
        Specify the main entry (e.g., main_package.main_py:main_func) < main_package.main_py:main_func
        Specify the port that docker will listen < 80
        Is private project? (y/n, default is y) < n
        Install .gitlab-ci.yml? (y/n, default is y) < n

    Python 2:
        Project Name (default is test) <
        Project Type (python/docs/web) < python
        Install travis? (y/n, default is y) < n
        Install docs? (y/n, default is y) < n
        To deploy as a service (y/n, default is y) <
        Choose a service type (flask/cli) < cli
        Choose a image (default is python:3.6) <
        Specify the main entry (e.g., the path to main.py) < main.py
        Image Port (default is None) < 80
        Is private project? (y/n, default is y) < n
        Install .gitlab-ci.yml? (y/n, default is y) <
        Need [build] Stage? (y/n, default is n) < n
        Need [test] Stage? (y/n, default is y) < n
        Need [review] Stage? (y/n, default is y) <
        Choose a image < hello
        Add Corresponding Stop Stage? (y/n, default is y) < n
        Only triggered in master branch? (y/n, default is n) <
        Triggered manually? (y/n, default is n) <
        Need [production] Stage? (y/n, default is y) < n
    """
    project = tmp_path_factory.mktemp("deploy")
    inputs = ["arch", "web"] + ["n", "n", "", "nginx"] + [""] * 5 + ["n"] * 4
    with simulate_stdin(*inputs):
        cli(skip_top=False, tar_dir=project)

    inputs = ["web_arch", "web"] + ["n"] + [""] * 8 + [
        "nginx"] + [""] * 2 + ["80", "n"] + [""] * 2 + [
        "nginx"] + ["n"] * 2 + ["", "nginx", ""] + ["n"] * 3
    with simulate_stdin(*inputs):
        cli(skip_top=False, tar_dir=project)

    inputs = ["python_arch", "python"] + [
        "n", "n", "", "flask", ""] + ["main_package.main_py:main_func", "80", "n", "n"]
    with simulate_stdin(*inputs):
        cli(skip_top=False, tar_dir=project)

    inputs = ["python_arch2", "python"] + [
        "n", "n", "", "cli", ""] + ["main_py", "80", "n", "", "n", "n", "", "hello", "", "n", "", "", "n"]
    with simulate_stdin(*inputs):
        cli(skip_top=False, tar_dir=project)
