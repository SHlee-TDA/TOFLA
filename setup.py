from setuptools import setup, find_packages

setup(
    name='TOFLA',
    version='0.1.0',
    author='Seong-Heon Lee',
    author_email='shlee0125@postech.ac.kr',
    description='Topologically Optimized Filtration Learning Algorithm for TDA on RGB images',
    long_description=open('README.md').read(),
    long_description_content_type='docs/markdown',
    url='https://github.com/SHlee-TDA/TOFLA',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'gudhi',
        'scikit-learn',
        'pytorch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
    keywords='topological data analysis, TDA, image processing, optimization',
)