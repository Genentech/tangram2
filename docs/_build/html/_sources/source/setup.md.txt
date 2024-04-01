## Installation
To install the TELEGRAPH pipeline, do:

### 1. Clone this repository

Clone your repository into a folder of your choice (`REPO_FOLDER`)

```sh
cd REPO_FOLDER
git clone git@ssh.code.roche.com:rb-aiml-cv-spatial/cci-explore/validation/pipeline.git
cd pipeline
```

### 2. Install using poetry

Since we ingest multiple methods with a complex set of dependencies, we decided
to use [poetry](https://python-poetry.org/) for packaging and dependency
management. Make sure you have poetry installed before proceeding.

Another recommendation is that you use our pre-defined conda environment and
install the package with poetry within this environment. To do this, do:

```sh
conda env create -f environment.yml
conda activate telegraph
```

If you are working in a remote server, you might have to load your
Anaconda/conda module, this might look something like:

```sh
ml load Anaconda3/2021.0
```


To circumvent dependency issues, we host "forked" versions of certain CCC
packages. These packages are deposited into Roche's gitlab server, meaning that
an extra layer of security exist and you need a username+password to access
these repositories. Fortunately, we've provided you with a config file that
takes care of all of this, all you need to run it with your gitlab username as
an argument.


```sh
. config.sh YOUR_GITLAB_USERNAME
poetry install
```
