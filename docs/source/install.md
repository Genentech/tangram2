# Installation
To install the `telegraph` pipeline, do:

## Clone the GitLab repository

Clone the `telegraph` [repository](https://code.roche.com/rb-aiml-cv-spatial/cci-explore/validation/pipeline) into a folder of your choice (`REPO_FOLDER`)

```sh
cd REPO_FOLDER
git clone git@ssh.code.roche.com:rb-aiml-cv-spatial/cci-explore/validation/pipeline.git telegraph
cd telegraph
```

## Install using poetry

Since we ingest multiple methods with a complex set of dependencies, we decided
to use [poetry](https://python-poetry.org/) for packaging and dependency
management. Make sure you have poetry installed before proceeding.

### Create and acitvate environment

We recommend you using a package manager to not break dependencies. We provide
an anaconda environment, to use this do:


```sh
conda env create -f environment.yml
conda activate telegraph
```

**Note**: If you are working on a remote server, you might have to load your
Anaconda/conda module, this might look something like:

```sh
ml load Anaconda3/2021.0
```

### Configure repository for package access

To circumvent dependency issues, we host "forked" versions of certain packages
packages. These packages are deposited into Roche's GitLab server, this means
you also have to run an additional configuration script.


```sh
. config.sh YOUR_GITLAB_USERNAME
poetry install
```

### [Optional] : Test for success
If you want to test whether `telegraph` has been successfully installed, you can do:

```python
python -c "import telegraph as tg"
```

If this command executes successfully, you are good to go.
