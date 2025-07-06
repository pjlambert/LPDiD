from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LPDiD",
    version="0.5.0",
    author="Peter John Lambert",
    author_email="p.j.lambert@lse.ac.uk",
    description="Local Projections Difference-in-Differences (LP-DiD) for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LPDiD",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "pyfixest>=0.18.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "joblib>=1.0.0",
        "wildboottest>=0.1.0",
    ],
)
