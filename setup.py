import setuptools
from distutils.core import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyFWI",
    version="0.0.0.3",
    author="Amir H. Mardan",
    author_email="mardan.ah69@gmail.com",
    description="A package for exploration seismology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmirMardan/seismic",
    project_urls={
        "Bug Tracker": "https://github.com/AmirMardan/seismic/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)