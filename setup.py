from setuptools import setup, find_packages

setup(
    name='PyEpiDAG',
    version='2.5.1',
    packages=find_packages(),
    url='https://github.com/TimeWz667/PyEpiDAG',
    license='MIT',
    author='TimeWz',
    author_email='TimeWz667@gmail.com',
    description='Epidemiological inference with DAG',
    install_requires=['pandas', 'networkx', 'astunparse', 'numpy', 'scipy', 'matplotlib']
)
