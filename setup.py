from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="apex-black-box-v40",
    version="4.0.0",
    author="Sib-asian",
    description="Oracle Engine per pronostici calcio live — Apex Black Box v4.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sib-asian/apex-black-box-v40-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "Flask>=2.3.0",
        "flask-cors>=4.0.0",
        "streamlit>=1.32.0",
        "plotly>=5.18.0",
    ],
)