from setuptools import setup, find_packages

setup(
    name='apodock',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas',
        'torch',
        'torch-geometric',      
    ],
    author='Ding Luo, Xiaoyang Qu, Dexin Lu, Yiqiu Wang, Lina Dong, Binju Wang and Xin Dai',
    description='ApoDock is a modular docking paradigm that combines machine learning-driven conditional side-chain packing based on protein backbone and ligand information with traditional sampling methods to ensure physically realistic poses.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/empyriumz/ApoDock_public',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)