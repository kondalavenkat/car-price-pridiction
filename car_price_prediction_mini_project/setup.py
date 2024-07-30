from setuptools import setup, find_packages

def get_requirements(filepath):
    """
    This function is responsible for getting the requirements from the filepath and feed it to the model
    """
    requirements = []
    with open(filepath, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", '') for req in requirements]
    
    if '-e .' in requirements:
        requirements.remove('-e .')
        
    return requirements

setup(
    name="CarValueML",
    version="0.0.1",
    author="venkata kondalarao kanuri",
    author_email="kondalaraokanuri2002@gmail.com",
    packages=find_packages,
    install_requires=get_requirements('requirements.txt')
)
