from setuptools import setup, Command
from subprocess import call
from os import path
import glob


class CoverageCommand(Command):
    description = 'run test code coverage'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        call(['coverage', 'run', '-m', 'unittest', 'discover', 'tests'])
        print()
        call(['coverage', 'report', '-m'])


class ExamplesCommand(Command):
    description = 'run examples and produce HTML reports'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import pweave

        for example_file in glob.iglob('examples/**/*.md', recursive=True):
            print(example_file)
            pweave.weave(example_file, doctype='md2html')


with open('README.md') as f:
    long_description = f.read()


setup(
    name='dsntnn',
    version='0.5.0',
    author='Aiden Nibali',
    description='PyTorch implementation of DSNT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    packages=['dsntnn'],
    test_suite='tests',
    cmdclass={
        'coverage': CoverageCommand,
        'examples': ExamplesCommand,
    },
    install_requires=[
        'torch>=0.4',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ]
)
