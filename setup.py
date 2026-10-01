from setuptools import find_packages, setup

setup(
    name="facial-recognition-pipeline",
    version="0.1.0",
    description="A deep learning-based facial recognition pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.9",
)
