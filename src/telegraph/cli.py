import os.path as osp
from os import makedirs

import click
import yaml


# create click group
@click.group()
def cli():
    pass


# add subcommand 'eval' to group
@cli.command("eval")
@click.option("-cf", "--config", help="configuration file", required=True)
@click.option("-od", "--out_dir", help="output directory", required=True)
@click.option("-sv", "--save_mode", help="save_outputs", is_flag=True, default=False)
@click.option("-db", "--debug", help="debug", is_flag=True, default=False)
def eval(config: str, out_dir, save_mode: bool = False, debug: bool = True):
    import telegraph.evaluation as ev

    """Execute and/or evaluate CCC workflows"""

    # check if the config path is a directory
    if osp.isdir(config):
        # if a directory treat all yaml files in directory as
        # relevant config files
        _config = [c for c in config if c.endswith(("yml", "yaml"))]
        # if no yaml files in directory, print error
        assert len(_config) >= 1, "no config files found in the directory: {}".format(
            config
        )
    elif osp.isfile(config):
        # make sure path is yaml file
        assert config.endswith(
            ("yml", "yaml")
        ), "config file needs to be in a yaml format"
        # put config path in list format
        _config = [config]
    else:
        NotImplementedError

    # iterate over config files
    for cf in _config:

        # load config file
        with open(cf, "r") as f:
            design = yaml.safe_load(f)

        # if single config file
        if len(_config) == 1:
            cf_out_dir = out_dir

        # if multiple config files
        else:
            # set output directory name based on config name
            cf_base = ".".join(osp.basename(cf).split(".")[0:-1])
            cf_out_dir = osp.join(out_dir, cf_base)

        # create output directory (if necessary)
        makedirs(cf_out_dir, exist_ok=True)

        # run workflow using config file
        ev.run.run(design, cf_out_dir, save_mode)


# add subcommand 'eval' to group
@cli.command("test")
def test():
    print("The CLI for telegraph is working")


if __name__ == "__main__":
    cli()
