from setuptools import setup, find_packages

setup(
    name='PyEpiDAG',
    version='2.1.1',
    packages=find_packages(),

    # ['epidag',
    #          'epidag.data'
    #          'epidag.factory',
    #          'epidag.bayesnet',
    #          'epidag.simulation',
    #          'epidag.fitting',
    #          'epidag.fitting.alg',
    #          'epidag.causality'],
    url='https://github.com/TimeWz667/PyEpiDAG',
    license='MIT',
    author='TimeWz',
    author_email='TimeWz667@gmail.com',
    description='Epidemiological inference with DAG',
    install_requires=['pandas', 'networkx', 'astunparse', 'numpy', 'scipy', 'matplotlib']
)
