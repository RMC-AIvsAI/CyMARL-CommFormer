import sys
from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "The CybORG repo is designed to work with Python 3.7 and greater." \
    + "Please install it before proceeding."

with open('version.txt') as f:
    CYBORG_VERSION = f.read()[:-1]

setup(
    name="CybORG",
    version=CYBORG_VERSION,
    description="A Cyber Security Research Environment",
)
