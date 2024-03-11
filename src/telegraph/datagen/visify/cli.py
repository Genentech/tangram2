import argparse as arp
import os.path as osp
from os import makedirs

import anndata as ad
import utils as ut
import visify


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
        "-sdist",
        "--spot_distance",
        type=float,
        default=None,
        help="distance between spots (both dimensions)",
    )
    aa(
        "-ecps",
        "--expected_cells_per_spot",
        type=int,
        default=None,
        help='desired number of cells per spot. Will override "spot diameter',
    )

    aa("-sdia", "--spot_diameter", type=float, default=55, help="spot diameter")
    aa(
        "-skey",
        "--spatial_key",
        type=str,
        default="spatial",
        help="key for spatial coordinates in anndata objects",
    )
    aa(
        "-plt",
        "--plot",
        default=False,
        action="store_true",
        help="save visuals of the synthetic data",
    )
    aa(
        "-rmp",
        "--return_mappable",
        default=False,
        action="store_true",
        help="return anndata object of all mappable cells",
    )
    aa(
        "-dws",
        "--downsample",
        default=False,
        action="store_true",
        help="downsample data",
    )
    aa(
        "-plw",
        "--p_lower",
        type=float,
        default=0.85,
        help="upper bound for downsampling of transcripts at spot",
    )
    aa(
        "-pup",
        "--p_upper",
        type=float,
        default=1.0,
        help="upper bound for downsampling of transcripts at spot",
    )

    aa(
        "-ai",
        "--add_indicator",
        default=False,
        action="store_true",
        help="add indicator of mappable cells to original anndata object",
    )
    aa(
        "-mn",
        "--p_mul_noise",
        default=None,
        type=float,
        nargs="+",
        help="proportion of multinomial noise. sevaral proportion values can be provided",
    )

    aa("-tg", "--tag", default=None, type=str, help="tag to add to output names")

    args = prs.parse_args()

    makedirs(args.out_dir, exist_ok=True)

    ut._check_vals(args.p_lower, "p_lower", 0, 1)
    ut._check_vals(args.p_upper, "p_upper", 0, 1)
    ut._check_vals(args.p_mul_noise, "p_upper", 0, 1)

    if args.tag is None:
        tag = ""
    else:
        tag = args.tag + "_"

    for adata_pth in args.adata:
        adata = ad.read_h5ad(adata_pth)
        prms = dict(
            spot_dist=args.spot_distance,
            expected_cells_per_spot=args.expected_cells_per_spot,
            spot_diameter=args.spot_diameter,
            spatial_key=args.spatial_key,
            downsample=args.downsample,
            return_mappable=args.return_mappable,
            add_indicator=args.add_indicator,
            p_lwr=args.p_lower,
            p_upr=args.p_upper,
            p_mul=args.p_mul_noise,
        )

        vis_res = visify.visify(adata, **prms)
        prms["og_adata_pth"] = adata_pth
        vs_ad = vis_res["vs"]

        vs_ad.uns["visify_params"] = prms

        vs_out_pth = osp.join(
            args.out_dir,
            tag + osp.basename(adata_pth).replace(".h5ad", "_visified.h5ad"),
        )

        vs_ad.write_h5ad(vs_out_pth)

        if args.return_mappable:
            mp_ad = vis_res["mp"]
            mp_out_pth = osp.join(
                args.out_dir,
                tag + osp.basename(adata_pth).replace(".h5ad", "_mappable.h5ad"),
            )
            mp_ad.write_h5ad(mp_out_pth)

        if args.add_indicator:
            adata = vis_res["adata"]
            adata.write_h5ad(adata_pth)

        if args.plot:
            fig, ax = ut.plot_visify(
                adata, vs_ad, spatial_key=args.spatial_key, feature=None
            )

            pl_out_pth = osp.join(
                args.out_dir,
                tag + osp.basename(adata_pth).replace(".h5ad", "_visify_plot.png"),
            )
            fig.savefig(pl_out_pth)


if __name__ == "__main__":
    main()
