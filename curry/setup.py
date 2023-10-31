from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name='curry',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={'console_scripts': ['curry=curry.main:main']},
    author="John McLevey, University of Waterloo",
    description="Curry is the GOAT, running point on your projects... ;)",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
)
