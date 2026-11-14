import click


@click.command()
@click.argument("name")
def main(name: str) -> None:
    click.echo(f"Hello {name}")


if __name__ == "__main__":
    main()
