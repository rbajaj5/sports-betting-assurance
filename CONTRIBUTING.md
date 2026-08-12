# Contributing

Contributions are welcome when they improve the assurance framework without
turning the project into a pick-selling or sportsbook-promotion product.

## Local workflow

1. Create a virtual environment.
2. Install the project with `python -m pip install -e ".[dev]"`.
3. Add or update tests for behavioral changes.
4. Run `python -m pytest` and `ruff check .`.
5. Open a focused pull request explaining the assurance failure being addressed.

Please use synthetic or openly licensed example data. Do not commit sportsbook
credentials, personal betting histories, or proprietary feeds.

