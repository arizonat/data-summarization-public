from setuptools import setup, find_packages

setup(
    name="detection-coresets",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your project dependencies here
        # "numpy>=1.20.0",
        # "pandas>=1.3.0",
        "torch",
        "torchvision"
    ],
    author="Levi Cai",
    author_email="cail@mit.edu",
    description="Core-sets library for detection tasks",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)