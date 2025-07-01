from setuptools import setup

def parse_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    install_requires=parse_requirements("reqs/requirements.txt"),
    extras_requires ={'cuda':parse_requirements("reqs/cuda-requirements.txt")},
)
