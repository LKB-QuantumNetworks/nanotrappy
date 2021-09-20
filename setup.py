from setuptools import _install_setup_requires, setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="nanotrappy",
    version="0.1",
    description="Package for computing dipole traps around nanostructures",
    long_description=readme(),
    url="https://github.com/LKB-QuantumNetworks/nanotrappy",
    author="Jérémy Berroir & Adrien Bouscal",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.npy"]},
    #   install_requires=[
    #       'numpy','scipy','simpy','matplotlib','json5','datetime','ARC-Alkali-Rydberg-Calculator','legume-gme'
    #   ],
    install_requires=requirements,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
    include_package_data=True,
)

