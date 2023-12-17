from setuptools import setup, find_packages

setup(
    name="CyMARL",
    version="1.0",
    packages=find_packages(include=['pymarl2', 'CybORG']),
    install_requires=[
        'docutils',
        'gym==0.26.0',
        'imageio', 
        'matplotlib',
        'networkx',
        'numpy',
        'paramiko',
        'prettytable',
        'probscale', 
        'protobuf==3.19.5',
        'pygame', 
        'pytest',
        'pyyaml==5.3.1', 
        'sacred==0.8.2',
        'scikit-learn==0.24.2',
        'scipy',
        'seaborn',
        'snakeviz',
        'tensorboard-logger',
    ]
)
