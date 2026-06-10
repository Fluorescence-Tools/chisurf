import os
import sys
import pathlib
from setuptools import setup, find_packages


setup(
    name="chisurf",
    version=os.environ.get('CHISURF_VERSION', '26.dev0'),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
