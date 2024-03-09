from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            if line and not line.startswith("-e ."):
                requirements.append(line)
    
    return requirements

setup(
    name="Classification",
    version="0.0.1",
    author="Oshionwu Victor",
    author_email="omolevictor97@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)
