# 文档自动生成说明[未完成，部分可用]
1. 配置所需包
在 docs 目录下运行
```sh
pip install -r requiremnets.txt
```

2. 自动配置[可省略]
在 docs 目录下运行
```sh
sphinx-quickstart
```
设置工程名等信息

2. 补充信息[可省略]
在 docs 目录下
修改 conf.py, 添加以下信息
    1. 添加路径
    ```python
    sys.path.insert(0, os.path.abspath('../'))
    ```
    2. 添加配置
    ```python
    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.intersphinx',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax',
        # 'sphinx_gallery.gen_gallery',
        'nbsphinx',
        'IPython.sphinxext.ipython_console_highlighting',
        'IPython.sphinxext.ipython_directive',
    ]
    ```
    3. 切换主题
    找到
    ```python
    html_theme = "sphinx_rtd_theme"
    ```
    换为
    ```python
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    ```
3. 生成rst数据
在 docs 目录下运行
```sh
sphinx-apidoc -f -o source/ ../src
```

4. 生成网页文档
在 docs 目录下运行
```sh
make html
```
