import click
from .config import Config

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option("--skeleton_csv", default=None, type=click.Path())
@click.option("--skeleton_config", default=None, type=click.Path())
@pass_config
def cli(config: Config, skeleton_csv, skeleton_config):
    config.skeleton.from_toml(skeleton_config)
    config.skeleton.path = skeleton_csv


@cli.command()
@click.option("--ouput_file", default="-", type=click.Path())
@pass_config
def query_jans_segmentation(config, output_file):
    from .scripts import query_jans_segmentation

    query_jans_segmentation(config, output_file)
