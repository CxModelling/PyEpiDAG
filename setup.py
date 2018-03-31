from setuptools import setup

setup(
    name='PyEpiDAG',
    version='1.2',
    packages=['epidag', 'epidag.bayesnet', 'epidag.factory', 'epidag.simulation', 'epidag.fitting', 'epidag.causality'],
    url='https://github.com/TimeWz667/PyEpiDAG',
    license='MIT',
    author='TimeWz',
    author_email='TimeWz667@gmail.com',
    description='Epidemiological inference with DAG', install_requires=['pandas', 'networkx', 'astunparse',
                                                                        'numpy', 'scipy', 'matplotlib']
)
