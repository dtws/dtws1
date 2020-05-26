#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "colormath",
    "folium",
    "geojson",
    "geopy",
    "h3",
    "matplotlib",
    "numpy",
    "pandas",
    "pyproj",
    "tilemapbase"
]

setup_requirements = []

test_requirements = []

setup(
    author="Yasunori Horikoshi",
    author_email='yasunori_horikoshi@datawise.co.jp',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="tools for data science",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dtws1',
    name='dtws1',
    packages=find_packages(include=['dtws1', 'dtws1.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/hotoku-dtws/dtws1',
    version='0.3.1',
    zip_safe=False,
)
