# Visify Tool

This is a small tool `visify` that takes single cell resolution spatial data as input an generates Visium-like data as output.


## Process

1. A grid with `spot_distance` length units between each node (hereafter: spot) will be generated over the spatial domain that the original data inhabits
2. All cells within a distance `spot_diameter /2` to a spot will be _assigned_ to that spot.
3. The "raw" gene expression profile of each spot is the _sum_ of the transcripts of all cells assigned to that spot. This will be found in the `X` attribute of your output `anndata` object.

If you specify no further options in the CLI, you will end up with an `anndata` object that looks something like:

```
AnnData object with n_obs × n_vars = 4125 × 313
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'visify_cell_map_og', 'visify_params'
    obsm: 'spatial'
```

The `adata.uns['visify_cell_map_og']` indicates which cell in the original `anndata` object that maps to which spot in the "visififed" data.

**Additional options:**

Use the `-h/--help` option for more information about additional arguments to the script. Some arguments require a bit more of an explanation, these are outlined below:

* `--p_mul_noise` : proportion of multinomial noise to add. A list of proportion values can be provided. The "noisy data" will be in `adata.layers['mul_noise_{p_mul_noise}']` as a sparse matrix.
* `--downsample` : downsample the observed transcripts at spot `s` to `p_s ~ Unif(--p_lower,--p_upper)` of the observed transcripts. The downsampled data will be in `adata.layers['downsample_{p_lower}_{p_upper}']`' as a sparse matrix.
* `--add_indicator` will add an indicator (`True` or `False`) to each cell in the original `anndata` object indicating whether it's included in the synthetic data or not. Some cells are excluded since they reside "between" spots. This indicator is found in  `adata.obs['mappable']` of the original `anndata` object.
* `--return_mappable` return an anndata object consisting of only the cells from the original `anndata` object that are included in the visified data, see `--add_indicator` for an explanation of why not all cells are included.
* `--expected_cells_per_spot` will override any value for `spot_radius` and try to create a grid where the spot size results in, on avearage, the value specified together with this flag.


### Example

Running the command: `python3 cli.py -ad ../../data/xenium_bc/curated/bc_rep_1.h5ad -sdist 100 -sdia 55 -rmp -mn 0.5 -od ~/dump/visify_test/ -tg 'qx' -dws -plt`

Will render the following output files:

```
qx_bc_rep_1_mappable.h5ad
qx_bc_rep_1_visified.h5ad
qx_bc_rep_1_visify_plot.png
```

If we inspect them further we have,

For `qx_bc_rep_1_visified.h5ad`:

```
AnnData object with n_obs × n_vars = 4125 × 313
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'visify_cell_map_mp', 'visify_cell_map_og', 'visify_params'
    obsm: 'spatial'
    layers: 'downsampled_0.85_1.00', 'mul_noise_0.50'
```

For `qx_bc_rep_1_mappable.h5ad`:

```
AnnData object with n_obs × n_vars = 39945 × 313
    obs: 'cell_id', 'x_centroid', 'y_centroid', 'transcript_counts', 'control_probe_counts', 'control_codeword_counts', 'total_counts', 'cell_area', 'nucleus_area'
    var: 'gene_ids', 'feature_types', 'genome'
    uns: 'visify_cell_map'
    obsm: 'spatial'
```

and for `qx_bc_rep_1_visify_plot.png`:

![visify example](../../imgs/visify/bc_example.png)
