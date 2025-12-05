# Tangram2

### Description
To be filled in


### Installation
To install `tangram2`, you can use the following command:

```bash
# Clone the repository
git clone https://github.com/Genentech/tangram2.git

# Change to the tangram2 directory
cd tangram2

# Create the environment using conda/mamba/micromamba
# Substitute {x} with your environment (eg macosx, linux, windows)
micromamba env create --file envs/{x}_environment.yml

# Activate the environment
micromamba activate tangram2

# Install the package
pip install .[cuda]
```
<!--  If you're on a machine that doesn't support CUDA (e.g., MacOSX) replace the last line with `pip install .` and you'll have a reduced set of dependencies. -->

For the environment, the `X` prefix represents the OS you are using, e.g., `linux`, `macos`, or `windows`.


### Tangram2 Paper
You can find all the notebooks used to generate the figures in the paper in the `analysis` directory. The notebooks are organized by figure number, and each notebook contains the code used to generate the corresponding figure in the paper.


### Contributing/Developing

Please make sure to use `pre-commit` when committing your changes. You can install `pre-commit` using the following command:

```bash
pip install pre-commit
```
Then, run the following command to install the pre-commit hooks:

```bash
pre-commit install
```
This will ensure that your code follows the project's coding standards and style guidelines before you commit your changes.

<!-- Please make sure to follow the [contribution guidelines](CONTRIBUTING.md) when contributing to this project. -->
