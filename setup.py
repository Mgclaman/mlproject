from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path: str) -> list[str]:
setup(
    name="mlprojects",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)