"""Allow ``python -m betguard`` to invoke the CLI."""

from betguard.cli import main

raise SystemExit(main())
