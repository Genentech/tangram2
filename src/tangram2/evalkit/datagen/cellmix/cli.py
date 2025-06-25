import argparse as arp
import os.path as osp
from os import makedirs

import anndata as ad
import cellmix


def main():
    prs = arp.ArgumentParser()

    aa = prs.add_argument

    aa("-ad", "--adata", required=True, nargs="+", help="path to anndata object(s)")
    aa(
        "-od",
        "--out_dir",
        required=True,
        type=str,
        help="output directory, if not exists it will be created",
    )
    aa(
        "-ecps",
        "--expected_cells_per_spot",
        type=int,
        default=10,
        help="desired number of cells per spot",
    )

    aa(
        "-etps",
        "--expected_types_per_spot",
        type=int,
        default=3,
        help="desired number of types per spot",
    )

    aa(
        "-lbl",
        "--label_col",
        type=str,
        default=None,
        help="name of label column",
        required=True,
    )

    aa(
        "-dws",
        "--downsample",
        default=None,
        help="downsample data",
        nargs="+",
    )

    aa(
        "-ns",
        "--n_spots",
        help="number of spots",
        required=True,
        type=int,
    )

    aa(
        "-es",
        "--encode_spatial",
        default=False,
        action="store_true",
        help="make the spots coordinates encode spatial information",
    )

    aa(
        "-nsg",
        "--n_spatial_gradient",
        default=None,
        type=float,
        help="how many spatial gradient genes should be added",
    )

    aa(
        "-nint",
        "--n_interactions",
        default=None,
        type=float,
        help="how many interactions should be added",
    )

    aa(
        "-esz",
        "--effect_size",
        default=10,
        type=int,
        help="number of downstream effects associated with each interaction",
    )

    aa("-tg", "--tag", default=None, type=str, help="tag to add to output names")

    args = prs.parse_args()

    makedirs(args.out_dir, exist_ok=True)

    if args.tag is None:
        tag = ""
    else:
        tag = args.tag + "_"

    for adata_pth in args.adata:

        adata = ad.read_h5ad(adata_pth)

        prms = dict(
            n_spots=args.n_spots,
            n_cells_per_spot=args.expected_cells_per_spot,
            n_types_per_spot=args.expected_types_per_spot,
            label_col=args.label_col,
            downsample=args.downsample,
            encode_spatial=args.encode_spatial,
            n_spatial_grad=args.n_spatial_gradient,
            n_interactions=args.n_interactions,
            effect_size=args.effect_size,
        )

        ad_sp, ad_sc = cellmix.cellmix(adata, **prms)
        prms["og_adata_pth"] = adata_pth

        ad_sp.uns["cellmix_params"] = prms

        sp_out_pth = osp.join(
            args.out_dir,
            tag + osp.basename(adata_pth).replace(".h5ad", "_cellmixed.h5ad"),
        )

        ad_sp.write_h5ad(sp_out_pth)

        sc_out_pth = osp.join(
            args.out_dir,
            tag + osp.basename(adata_pth).replace(".h5ad", "_mappable.h5ad"),
        )
        ad_sc.write_h5ad(sc_out_pth)


if __name__ == "__main__":
    main()
