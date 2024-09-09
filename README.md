# Python Project Template

This project template serves as a robust foundation for Python projects, promoting best practices and streamlining development workflows. It comes pre-configured with essential tools and features to enhance the development experience.

## Tools Included

- [PDM](https://pdm-project.org) for efficient dependency management.
- [Ruff](https://docs.astral.sh/ruff) for comprehensive linting and code formatting.
- [Pytest](https://docs.pytest.org) for running tests and ensuring code reliability.
- [pre-commit](https://pre-commit.com) for managing pre-commit git hooks ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), ruff check, ruff format)
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci) for continuous integration.
- [Pydocstyle](https://www.pydocstyle.org) for validating docstring styles, also following the [Google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
- [Mypy](https://mypy-lang.org) for static type checking.


## Usage

Key commands for effective project management:

- `pdm install` - Installs all project dependencies.
- `pdm add <package>` - Adds a new dependency to the project.
- `pdm l` - Runs linting, formatting, and mypy checks.
- `pdm test` - Executes tests located in the tests directory.
- `pdm run <command>` - Runs the specified command within the virtual environment.

## CI/CD

The project uses our [GitLab CI/CD templates](https://gitlab.ics.muni.cz/rationai/digital-pathology/templates/ci-templates) to automate the linting and testing processes. The pipeline is triggered on every merge request and push to the default branch.
