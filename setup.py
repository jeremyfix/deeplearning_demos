from setuptools import setup

setup(
    name='deeplearning_demos',
    version='0.1.0',
    author='Jeremy Fix',
    author_email='jeremy.fix@gmail.com',
    packages=['deeplearning_demos'],
    license='LICENSE.txt',
    description='Some scripts to easily raise up deep learning demos',
    entry_points={
        'console_scripts': ['segmentation_server=deeplearning_demos.segmentation_server:main']
    }
)
