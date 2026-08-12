from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="augagent",
    version="1.0.0",
    author="Augmencord Research",
    author_email="research@augmencord.com",
    description="An enterprise-grade autonomous ReAct agent framework with token streaming and hierarchical orchestration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/augmencord/augagent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "chromadb>=0.4.0",
    ],
    python_requires=">=3.10",
)
