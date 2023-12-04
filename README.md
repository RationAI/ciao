# Python project template

This project template serves as a robust foundation for Python projects, promoting best practices and streamlining development workflows. It comes preconfigured with essential tools and features to enhance the development experience.

## Tools Included

- [PDM](https://pdm-project.org) for efficient dependency management.
- [Ruff](https://docs.astral.sh/ruff) for comprehensive linting and code formatting.
- [Pytest](https://docs.pytest.org) for running tests and ensuring code reliability.
- [pre-commit](https://pre-commit.com) for managing pre-commit git hooks ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), ruff check, ruff format)
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci) for continuous integration.
- [Pylint](https://pylint.readthedocs.io) for static code analysis, adhering to the [Google style](https://google.github.io/styleguide/pyguide.html).
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

Two defined pipelines enhance project development:

1. **Synchronization pipeline(`ci/template` branch):**
    - Triggered by a scheduler for the ci/template branch.
    - Handles [synchronization](#synchronizing-the-template) with the template repository.

2. **Project Pipeline (Main branch and Merge Requests):**
    - Triggered by any push to the main branch or any merge request to the main branch.
    - Consists of the following stages:
        - `lint` - Runs linting, formatting, and mypy checks (ruff check, ruff format, mypy).
        - `test` - Executes tests using Pytest.


## Synchronizing the Template

For seamless integration with the template repository:

1. Create a protected branch named `ci/template`.
2. [Schedule a pipeline](https://docs.gitlab.com/ee/ci/pipelines/schedules.html) for the `ci/template` branch to automatically synchronize with the template repository.
3. Ensure the `ci/template` branch tracks the last synchronized commit.
4. The synchronization pipeline will:
    - Check out the last synchronized commit.
    - Merge the latest commit from the template repository, prioritizing template changes in case of conflicts.
    - Commit and push the changes to the `ci/template` branch.
    - Create a merge request to the default branch.



> **Note:**
> Resolve any conflicts manually, ensuring the **last synchronized commit** is retained in the ci/template branch to prevent reappearing conflicts during subsequent synchronizations.
