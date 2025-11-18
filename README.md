# Python Project Template

This project template serves as a robust foundation for Python projects, promoting best practices and streamlining development workflows. It comes pre-configured with essential tools and features to enhance the development experience.

## Tools Included

- [uv](https://docs.astral.sh/uv/) for efficient dependency management.
- [Ruff](https://docs.astral.sh/ruff) for comprehensive linting and code formatting.
- [Pytest](https://docs.pytest.org) for running tests and ensuring code reliability.
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci) for continuous integration.
- [Pydocstyle](https://www.pydocstyle.org) for validating docstring styles, also following the [Google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
- [Mypy](https://mypy-lang.org) for static type checking.


## Usage

Key commands for effective project management:

- `uv sync` - Installs all project dependencies.
- `uv add <package>` - Adds a new dependency to the project.
- `uv run ruff check` - Runs linting.
- `uv run ruff format` - Runs formatting
- `uv run mypy .` - Runs mypy.
- `uv run pytest tests` - Executes tests located in the tests directory.
- `uv run <command>` - Runs the specified command within the virtual environment.

## CI/CD

The project uses our [GitLab CI/CD templates](https://gitlab.ics.muni.cz/rationai/digital-pathology/templates/ci-templates) to automate the linting and testing processes. The pipeline is triggered on every merge request and push to the default branch.
