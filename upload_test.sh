#!/bin
twine check dist/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*