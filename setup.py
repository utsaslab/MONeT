import setuptools

setuptools.setup(
    name="monet_memory_optimized_training",
    version="0.0.1",
    description="Memory Optimized Network Training Framework",
    url="https://github.com/philkr/lowrank_conv",
    packages=setuptools.find_packages(include = ['monet', 'monet.*', 'models', 'checkmate', 'gist']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
