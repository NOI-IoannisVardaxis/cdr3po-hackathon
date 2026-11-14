# her2 stopping rule

Multi-objective antibody design pipeline for trastuzumab/HER2: affinity, epitope preservation and thermostability, with calibrated filters and an explicit stopping rule.

## Development

Install `uv` if you don't have it installed already:

```bash
curl --location \
     --silent  \
     --show-error \
     --fail \
     https://astral.sh/uv/install.sh | sh
```

See also [installing uv]

Clone the [repository][repo] & install the dependencies.

```bash
git clone git@github.com:NOI-IoannisVardaxis/her2-stopping-rule.git
cd her2-stopping-rule
# Setup python virtualenv & dependencies
make dev
# Run tests
make test
# Run lints
make lint
# Check if we're able to package the wheel
make wheel
```

### Dependency management
[uv] is used for dependency management. The workflow if you need to add a dependency is:

1. Run `uv add package` for production dependencies, or `uv add --dev package` for dev dependencies.
2. The dependencies you add are appended to [pyproject.toml] and locked down in [uv.lock] automatically. Add these changes to git.

#### Running with uv

Using `uv run` to execute python [scripts] ensures that dependencies are managed declaratively without manually managing python environments. `uv run` always ensures that the dependencies specified in the `pyproject.toml` file in your current working directory are installed before running.

[repo]: https://github.com/NOI-IoannisVardaxis/her2-stopping-rule
[docker]: https://docs.docker.com/docker-for-mac/install/
[uv]: https://github.com/astral-sh/uv
[tools]: https://docs.astral.sh/uv/guides/tools/
[scripts]: https://docs.astral.sh/uv/guides/scripts/
[installing uv]: https://docs.astral.sh/uv/getting-started/installation/
