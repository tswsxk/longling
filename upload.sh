#!/bin
# python setup.py check && python setup.py sdist && python setup.py register sdist upload
python3 setup.py sdist bdist_wheel && twine upload --repository-url https://test.pypi.org/legacy/ dist/*