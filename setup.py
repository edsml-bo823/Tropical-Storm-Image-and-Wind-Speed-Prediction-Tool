from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]
# The above section is inspired by reference[5]  
dependencies = read_requirements('requirements.txt')

setup(
    name='storm',
    version='0.1',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="Environment for running victor_functions model.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    author="ACDS Victor",
    author_email='acdsvictor2023@gmail.com',
    packages=find_packages(),
    install_requires=dependencies,
    
    classifiers=[
        # Choose appropriate classifiers from https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
