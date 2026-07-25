"""Compatibility entry point for the guided SyntheticCAD local app."""

from syntheticcad.local_app import AppHandler, build_parser, main, serve

__all__ = ["AppHandler", "build_parser", "main", "serve"]


if __name__ == "__main__":
    raise SystemExit(main())
