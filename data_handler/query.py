import click


@click.group()
@click.option("--skeleton_csv", default=None, type=click.Path())
@click.option("--skeleton_config", default=None, type=click.Path())
def cli(skeleton_csv, skeleton_config):
    pass


@cli.command()
def query_jans_segmentation():
    click.echo("querying jans segmentation!")
    return
    from .scripts import query_jans_segmentation

    query_jans_segmentation()
