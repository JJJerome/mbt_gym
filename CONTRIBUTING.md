# Welcome to the mbt_gym contributing guide

Thank you for considering contributing to `mbt_gym`!

Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) first to help us to maintain a friendly and helpful community.

## Pull requests (PRs)

Please feel free to open a Pull Request for any minor changes to the repository. For larger changes, please open an 
issue first to discuss with other users and maintainers of `mbt_gym`. If you are not familiar with creating a Pull 
Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

In particular, **please see the [roadmap.md](./roadmap.md) file**, for a list of desired additions that will be accepted.
Any appropriate tests will also always be accepted.

## Codestyle

We use [mypy](https://flake8.pycqa.org/en/latest/) as a static type checker, [Flake8](https://flake8.pycqa.org/en/latest/) to enforce PEP8 and [Black](https://black.readthedocs.io/en/stable/) to enforce consistent styling.

- Code will be automatically reformatted with: `invoke black-reformat`
- Styling and type checking tests can be run locally with: `invoke check-python`

## Tests

When adding new code to the `mbt_gym` code-base, please add test coverage wherever possible.
We use [unittest](https://docs.python.org/2/library/unittest.html) for unit testing. All unit tests can be run by 
calling `nose2` from the root directory.