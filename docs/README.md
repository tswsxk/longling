# longling documents with Sphinx
使用Sphinx生成longling文档

## 依赖
```sh
pip install sphinx sphinx-gallery nbsphinx sphinx_rtd_theme Image recommonmark ipython m2r
```

## 文档生成
### html
在docs 目录下执行
```sh
make html
```
即可在_build/html获得网页文件
### latex
在docs 目录下执行
```sh
make latex
```
在_build/latex目录下使用**Xelatex**进行编译