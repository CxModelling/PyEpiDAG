from setuptools import setup

setup(
    name='PyEpiDAG',
    version='1.2.1',
    packages=['epidag',
              'epidag.factory',
              'epidag.bayesnet',
              'epidag.simulation',
              'epidag.fitting',
              'epidag.causality'],
    url='https://github.com/TimeWz667/PyEpiDAG',
    license='MIT',
    author='TimeWz',
    author_email='TimeWz667@gmail.com',
    description='Epidemiological inference with DAG',
    install_requires=['pandas', 'networkx', 'astunparse', 'numpy', 'scipy', 'matplotlib'],
    python_requires='>=3.5'
)
