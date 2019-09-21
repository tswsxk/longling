#!/bin
# python setup.py.T check && python setup.py.T sdist && python setup.py.T register sdist upload
python3 setup.py.T sdist bdist_wheel && twine upload --repository-url https://upload.pypi.org/legacy/ dist/*