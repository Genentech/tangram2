# Tangram2

### Description
Tangram2 is a computational framework for learning cell–cell communication directly from single-cell and spatial transcriptomics data. Tangram2 is organized into three modules:

**Tangram2-mapping**
Aligns scRNA-seq profiles to spatial transcriptomics (SRT) by learning a probabilistic mapping from single cells to spots. This enables identification of co-localized cell populations while retaining transcriptome-wide coverage, and includes an integrated mapping mode to robustly handle both paired and unpaired scRNA/SRT datasets.

**Tangram2-CCC**
Builds on the mapping output to infer cell–cell communication effects. It fits a linear model that decomposes gene expression into intrinsic cell-type programs and interaction-driven components, yielding an “interaction tensor” of gene-level coefficients that quantify how each sender cell type modulates each receiver cell type’s genes. This allows systematic identification of interaction-induced transcriptional programs across tissues.

**Tangram2-evalkit**
Provides a synthetic data generation and benchmarking framework. Starting from real scRNA-seq with cell-type labels, it simulates Visium-like spot grids with configurable cell-type compositions, spatial structure, noise levels, and injected intercellular interactions. These datasets are then used to benchmark both mapping and CCC performance under controlled conditions.

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

Setting up the conda environment and completing the installation typically takes less than 10 minutes on a desktop computer.


### Tangram2 Paper
The Tangram2 paper is published in Inferring cellular communication through mapping cells in space using Tangram 2 at BioRxiv(DOI: https://doi.org/10.1101/2025.09.28.679077). You can find all the notebooks used to generate the figures in the paper in the `analysis` directory. The notebooks are organized by figure number, and each notebook contains the code used to generate the corresponding figure in the paper.


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
