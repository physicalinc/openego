from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="openego",
    version="0.1.0",
    author="Ahad Jawaid and Yu Xiang",
    author_email="contact@physical.inc",
    description="OpenEgo: A Large-Scale Multimodal Egocentric Dataset for Dexterous Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.openegocentric.com",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.900",
            "pre-commit>=2.0",
        ],
        "vis": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "matplotlib>=3.3.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.900",
            "pre-commit>=2.0",
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "matplotlib>=3.3.0",
        ]
    },
)