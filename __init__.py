"""
Mnemosyne Plugin for Hermes Agent
Entry point at repo root for `hermes plugins install` compatibility.
"""

# Delegate to the hermes_plugin package
from hermes_plugin import register

__all__ = ["register"]
