from setuptools import setup, find_packages

setup(
    name="fit",
    version="0.1.0",
    packages=find_packages(include=["core", "nn", "utils", "monitor", "train"]),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
    author="Łukasz Bielaszewski",
    author_email="lukaszbielaszewskibiz@gmail.com",
    description="A custom Machine Learning and MLOps library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Klus3kk/fit",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
