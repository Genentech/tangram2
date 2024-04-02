.. telegraph documentation master file, created by
   sphinx-quickstart on Thu Mar 28 11:34:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the home of telegraph
=====================================

``telegraph`` is a computational framework for cell cell communcation (CCC)
inference. It's designed to be highly modular and versatile while also providing
tools for inspection, comparison, evaluation, and interpretation of the output.
In contrast to many existing frameworks for CCC we've also put an emphasis on
multimodal datasets where we have both spatial and scRNA-seq data.

A sample of the tasks `telegraph` accomodates methods for are:
- Identification of interacting cells in a data set (grouping of cells)
- Identify the downstream events of a given ligand-receptor interaction
- Comparing interaction patterns across conditions
- Evaluate method performance on ground truth data
- Interpretation of the idenfified interaction events

We provide you with a flexible API designed for custom analysis in mind, if
you're familiar with frameworks such as `scanpy` this package has a very similar
interface.


Our framework is organized into multiple submodules, with the following short descriptions:

- Methods module (``methods``)
  - Run, chain, and combine multiple different SOTA methods
- Diagnostics module (``diagnostics``)
  - Inspect, Interpret, visualize, and compre outputs from each step
- Datagen module (``datagen``)
  - Generate synthetic data for HP tuning and evaluation
- Evaluation module (``evaluation``)
  - Evaluate and compare performance on ground truth
- Aggregation module (``aggregate``)
  - Aggregate results from multiple workflows to get a consensus output


**Note:** `telegraph` is still under development, keep this in mind as you use the package and please report any issues to our gitlab page.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/install
   modules/
