from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    '''
    This function will return the list of requirements.
    '''
    requirements:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()

                #ignoring the empty lines and -e .
                if requirement and requirement!='-e .':
                    requirements.append(requirement)
    except FileNotFoundError: #requirements.txt does not exist
        print("Requirements.txt file not found!")

setup(
    name = "BiteScore",
    version="0.0.1",
    author="Aditya Pande",
    author_email="aditya.p22@iiits.in",
    package = find_packages(),
    install_requires=get_requirements()
)

