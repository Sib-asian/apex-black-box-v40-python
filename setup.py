"""Setup script for Apex Black Box V40."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="apex-black-box",
    version="4.0.0",
    author="Apex Black Box",
    description="In-play football probabilistic engine with Poisson modelling and Oracle verdicts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sib-asian/apex-black-box-v40-python",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="football soccer betting poisson probability kelly criterion",
)
