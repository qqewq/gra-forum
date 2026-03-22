from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gra-forum",
    version="0.1.0",
    author="GRA Forum Team",
    author_email="contact@gra-forum.org",
    description="AI Agent Debate Orchestrator based on GRA principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gra-forum",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "black>=23.0.0", "mypy>=1.5.0"],
        "embeddings": ["sentence-transformers>=2.2.0"],
    },
)
