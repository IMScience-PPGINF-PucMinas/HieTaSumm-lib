from setuptools import setup, find_packages

setup(
    name='HieTaSumm',
    version='0.1.1',    
    license='MIT Licence',
    description='A hierarchical approach for video summarization',
    keywords='summarization hierarchy graph',
    author='Leonardo Vilela',
    author_email='leonardo.cardoso.794229@sga.pucminas.br',
    packages=find_packages(),
    package_data={'HieTaSumm': ['options.json']},
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        'torch',
        'Pillow',
        'higra',
        'tensorflow',
        'keras',
        'scipy',
        'networkx',
        'pathlib'
    ],
)