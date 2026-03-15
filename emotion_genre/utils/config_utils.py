"""Load YAML configs and apply as parser defaults (CLI overrides config)."""
import argparse
import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file. Returns a flat dict of key -> value."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _parser_dests(parser: argparse.ArgumentParser) -> set:
    """Return set of dest names for all actions that have one."""
    out = set()
    for action in parser._actions:
        if action.dest != "help":
            out.add(action.dest)
    return out


def apply_config(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> None:
    """Set parser defaults from config for keys that the parser knows. CLI args override when parsing.
    Also sets required=False for any action that gets a default from config, so config can supply
    values for arguments that are normally required on the CLI.
    """
    dests = _parser_dests(parser)
    overrides = {k: v for k, v in config.items() if k in dests}
    if overrides:
        for action in parser._actions:
            if action.dest in overrides and getattr(action, "required", False):
                action.required = False
        parser.set_defaults(**overrides)


def parse_args_with_config(parser: argparse.ArgumentParser, argv: list = None):
    """Parse args: if --config is in argv, load it and set parser defaults, then parse. CLI overrides config."""
    args_pre, _ = parser.parse_known_args(args=argv)
    config_path = getattr(args_pre, "config", None)
    if config_path and os.path.isfile(config_path):
        config = load_config(config_path)
        apply_config(parser, config)
    return parser.parse_args(args=argv)
