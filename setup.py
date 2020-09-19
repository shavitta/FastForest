import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-shavitta",
    version="0.0.1",
    author="Shavit Talman",
    author_email="shavitta@gmail.com",
    description="FastForest implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shavitta/FastForest.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)