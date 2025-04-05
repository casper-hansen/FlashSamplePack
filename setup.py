from setuptools import setup, find_packages

setup(
    name="flash_sample_pack",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numba",
        "transformers==4.50.3",
        "datasets==3.5.0",
        "accelerate==1.5.2",
        "numpy>=1.24.4,<=2.0.1",
        "flash-attn==2.7.4.post1",
    ],
    extras_require={
        "train": ["trl"],
        "dev": ["black", "pytest"],
    },
    python_requires=">=3.10",
    description="Sample packing for Flash Attention with HuggingFace Transformers",
    author="Casper Hansen",
    author_email="casperbh.96@gmail.com",
    url="https://github.com/casper-hansen/FlashSamplePack",
)
