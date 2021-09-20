---
template: overrides/main.html
title: Getting started
---

# Getting started

Nanotrappy can be directly used from [GitHub][1] by cloning the
repository on your device.

```
git clone https://github.com/jrmbr/nanotrappy.git
```

## Installation in a virtual environment (Recommended)

Compatibility has been tested for python 3.7, 3.8 and 3.9 on Windows and Linux. We recommend setting up a new virtual environment for installing `nanotrappy`, preferably with Anaconda, with Python version specified.
Numpy **has to be installed before** building the package.
As follows :

=== "Windows"

    First install a python distribution on you computer. Here we use the [Miniconda3][4] distribution (can be found easily using your favorite search engine).

    Then, if you want to be safe, create a new environment and install numpy and the package inside, in this order.

    ```
    conda create --name yourenvname python=3.8
    conda activate yourenvname
    conda install numpy
    ```

    If you don't have Anaconda, and using Windows, this also works.

    ```
    python -m venv yourenvname
    .\yourenvname\Scripts\activate
    pip install numpy
    ```

    Once the package is cloned you simply need to install it on your local device. For this, navigate into the package folder (while still in your virtual environment) and run:

    ```
    pip install .
    ```

    This will automatically install compatible versions of all dependencies, which most importantly contain [Legume][2] and [ARC-Alkali-Rydberg-Calculator][3].

    When you are finished working, you can leave the environment using:
    ```
    conda deactivate
    ```

    When you want to use the package, juste activate this environment again and you are good to go !

    ```
    conda activate yourenvname
    ```

=== "Linux"

    First install a python distribution on you computer. Here we use the [Miniconda3][4] distribution (can be found easily using your favorite search engine).

    Download the .sh file corresponding to your system configuration and install it.

    Then, if you want to be safe, create a new environment and install numpy and the package inside, in this order.

    ```
    conda create --name yourenvname python=3.8
    conda activate yourenvname
    conda install numpy
    ```

    Once the package is cloned you simply need to install it on your local device. For this, navigate into the package folder (while still in your virtual environment) and run:

    ```
    pip install .
    ```

    This will automatically install compatible versions of all dependencies, which most importantly contain [Legume][2] and [ARC-Alkali-Rydberg-Calculator][3].

    When you are finished working, you can leave the environment using:
    ```
    conda deactivate
    ```

    When you want to use the package, juste activate this environment again and you are good to go !

    ```
    conda activate yourenvname
    ```

[1]: https://github.com/jrmbr/nanotrappy
[2]: https://legume.readthedocs.io/en/latest/
[3]: https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/
[4]: https://docs.conda.io/en/latest/miniconda.html
