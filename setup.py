from setuptools import setup, find_packages
import os

__package_name__ = "vfa"

def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


__version__, cmdclass = get_version_and_cmdclass(__package_name__)


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# noinspection PyTypeChecker
setup(
    name=__package_name__,
    version=__version__,
    description="Vector Field Attention",
    long_description="Vector Field Attention for Deformable Image Registration",
    author="Yihao Liu",
    author_email="yliu236@jhu.edu",
    url="https://gitlab.com/yihao6/vfa",
    license="GNU General Public License v3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    include_package_data=True,
    keywords="surfaces reconstruction",
    entry_points={
        "console_scripts": [
            "vfa-run=vfa.main:main",
        ]
    },
    install_requires=install_requires,
    cmdclass=cmdclass,
)
