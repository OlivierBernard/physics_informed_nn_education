#!/usr/bin/env python
import builtins
import os
import pathlib

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = pathlib.Path(__file__).parent
builtins.__PINNs_TUTORIALS_SETUP__ = True


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):  # noqa: D103
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_long_description():  # noqa: D103
    text = open(PATH_ROOT / "README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install vital[dev]`
# From local copy of repo, use like `pip install ".[dev]"`
extras = {
    "dev": load_requirements(file_name="dev.txt"),
}


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="pinns-tutorials",
    version="0.0.1",
    description="Tutorials of physics-informed neural networks (PINNs)",
    author="Hang Jung Ling <hang-jung.ling@insa-lyon.fr>, Olivier Bernard <olivier.bernard@insa-lyon.fr>",
    url="https://github.com/OlivierBernard/physics_informed_nn_education",
    license="MIT",
    packages=["src"],
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    python_requires="~=3.10",
    setup_requires=[],
    install_requires=load_requirements(),
    extras_require=extras,
)
