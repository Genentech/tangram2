## Cellmix

<img src="../../imgs/cellmix/cellmix.png" alt="cellmix logo" width="200" height="200"/>


This is a small tool `cellmix` that takes scRNA-seq data and creates Visium-like mixed data (without spatial positions though)


## Process

1. We specify the number of expected cells per spot and the number of expected types per spot
2. We sample the number of cells per spot and the number of types per spot
3. For each spot `i`, we sample which cell types that should reside there
4. For each spot `i`, we sample `n_cells_i` from the cell types assigned to spot `i`

Note: a cell can only be assigned to a single spot (no replacement)


## Example

Running the command: `python3 ./cli.py -ad ../../data/CID4290_sc.h5ad -od example_folder -lbl 'celltype_minor' -ecps 10 -etps 4 -ns 15`

Will give two output files:

* `CID4290_sc_cellmixed.h5ad`
* `CID4290_sc_mappable.h5ad`

Where  `CID4290_sc_cellmixed.h5ad` will have on average 10 cells per spot, spread across (on average) 4 different types. There will be a total of 15 spots in the dataset. The cell types in the original data is given by the `.obs` columns `celltype_minor`. `CID4290_sc_mappable.h5ad` will hold all the cells in the original scRNA-seq dataset that were assigned to a spot.
