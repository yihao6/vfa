import setuptools
from pathlib import Path

# Dynamically read requirements.txt
here = Path(__file__).parent
requirements = here.joinpath("requirements.txt").read_text().splitlines()

setuptools.setup(install_requires=requirements)
