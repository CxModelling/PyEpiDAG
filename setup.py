from setuptools import setup

setup(
    name='PyEpiDAG',
    version='0.1',
    packages=['epidag', 'epidag.fitting', 'epidag.factory', 'epidag.bayesnet'],
    url='https://github.com/TimeWz667/PyEpiDAG',
    license='MIT',
    author='TimeWz',
    author_email='TimeWz667@gmail.com',
    description='Epidemiological inference with DAG', install_requires=['pandas', 'networkx', 'astunparse',
                                                                        'numpy', 'scipy', 'matplotlib']
)
