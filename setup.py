from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apex-black-box-v40",
    version="4.0.0",
    author="Sib-asian",
    description="Advanced data processing solution with Steam integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sib-asian/apex-black-box-v40-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "steampy>=1.1.4",
        "python-dotenv>=0.21.0",
        "PyYAML>=6.0",
    ],
)