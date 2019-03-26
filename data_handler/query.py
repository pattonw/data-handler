import click
from config import Config

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option("--skeleton_csv_path", default=None, type=click.Path())
@click.option("--skeleton_config", default=None, type=click.Path())
@pass_config
def cli(config: Config, skeleton_csv_path, skeleton_config):
    config.skeleton.from_toml(skeleton_config)
    config.skeleton.path = skeleton_csv_path


@cli.command()
@pass_config
def query_jans_segmentation(config):
    from .scripts import query_jans_segmentation

    query_jans_segmentation(config)
