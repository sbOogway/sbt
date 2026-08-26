"""CLI entry point — preserved for backward compatibility.

Usage::

    uv run python3 -m sbt --config config.toml --strategy overnight_drift
"""

if __name__ == "__main__":
    from .cli.main import main

    main()
