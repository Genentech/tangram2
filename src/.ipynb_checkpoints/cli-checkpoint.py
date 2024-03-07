import os.path as osp
from os import makedirs

import click
import yaml

import eval as ev


@click.group()
def cli():
    pass


@cli.command("eval")
@click.option("-cf", "--config", help="configuration file", required=True)
@click.option("-od", "--out_dir", help="output directory", required=True)
@click.option("-sv", "--save_mode", help="save_outputs", is_flag=True, default=False)
@click.option("-db", "--debug", help="debug", is_flag=True, default=False)
def eval(config: str, out_dir, save_mode: bool = False, debug: bool = True):
    """Execute and/or evaluate CCC workflows"""

    if osp.isdir(config):
        _config = [c for c in config if c.endswith(("yml", "yaml"))]
        assert len(config) >= 1, "no config files found in the directory: {}".format(
            config
        )
    else:
        _config = [config]

    for cf in _config:

        assert cf.endswith(("yml", "yaml")), "config file needs to be in a yaml format"

        with open(cf, "r") as f:
            design = yaml.safe_load(f)

        if len(_config) == 1:
            cf_out_dir = out_dir
        else:
            cf_base = ".".join(osp.basename(cf).split(".")[0:-1])
            cf_out_dir = osp.join(out_dir, cf_base)

        makedirs(cf_out_dir, exist_ok=True)

        ev.run.run(design, cf_out_dir, save_mode)


if __name__ == "__main__":
    cli()
