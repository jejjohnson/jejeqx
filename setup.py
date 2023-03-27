import os
import sys
import itertools
import codecs

from setuptools import find_namespace_packages, setup


try:
    import versioneer
except ImportError:
    # we have a versioneer.py file living in the same directory as this file, but
    # if we're using pep 517/518 to build from pyproject.toml its not going to find it
    # https://github.com/python-versioneer/python-versioneer/issues/193#issue-408237852
    # make this work by adding this directory to the python path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import versioneer


def read_requirements(filename):
    base = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(base, filename), "rb", "utf-8") as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


requirements = {
    "base": read_requirements("environments/requirements.txt"),
    "dev": read_requirements("environments/requirements-dev.txt"),
    "jlab": read_requirements("environments/requirements-jlab.txt"),
}

with open("README.md", encoding="utf8") as readme:
    long_description = readme.read()


setup(
    name="eqx-trainer",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_namespace_packages(include=["eqx_trainer"]),
    author="J. Emmanuel Johnson",
    author_email="jemanjohnson34@gmail.com",
    license="LICENSE",
    description="Lightweight trainer module for equinox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # project_urls={
    #     "Documentation": "https://jaxsw.readthedocs.io/en/latest/",
    #     "Source": "https://github.com/jejjohnson/jaxsw",
    # },
    install_requires=requirements["base"],
    python_requires=">=3.8",
    extras_require={
        **requirements,
        "all": list(itertools.chain(*list(requirements.values()))),
    },
    include_package_data=True,
    keywords=["python template"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Intended Audience :: Science/Research",
    ],
)
