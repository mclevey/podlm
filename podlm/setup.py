from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="odlm",
    version="0.0.10",
    packages=find_packages(),
    install_requires=["pandas"],
    extra_requires={"dev": ["twine>=4.0.2"]},
    python_requires=">=3.7",
    author="John McLevey, University of Waterloo",
    description="Project-specific utility functions and classes.",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
)
