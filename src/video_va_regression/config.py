"""
Configuration management for the video VA regression pipeline.

This module provides a lightweight configuration loader based on INI files.
Configuration sections are parsed into dictionaries with automatic type casting
for common scalar types (bool, int, float).

Supported configuration sections:
    - [paths]
    - [selection]
    - [scheduling]
    - [training]
    - [testing]

The configuration is intended to be loaded once and passed to training and
evaluation scripts.

This module does not provide a CLI interface.
"""


# =============================================================================
# Imports
# =============================================================================


from __future__ import annotations

import configparser
import os
from typing import Any, Dict


# =============================================================================
# Data Structures
# =============================================================================


class ConfigManager:
    """
    INI-based configuration loader with automatic type casting.

    Attributes:
        config_path (str): Path to the configuration file.
        selection (Dict[str, Any]): Parsed [selection] section.
        scheduling (Dict[str, Any]): Parsed [scheduling] section.
        training (Dict[str, Any]): Parsed [training] section.
        testing (Dict[str, Any]): Parsed [testing] section.
    """

    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize and load the configuration.

        Args:
            config_path (str): Path to the INI configuration file.
        """
        self.config_path = config_path

        self.paths: Dict[str, Any] = {}
        self.selection: Dict[str, Any] = {}
        self.scheduling: Dict[str, Any] = {}
        self.training: Dict[str, Any] = {}
        self.testing: Dict[str, Any] = {}

        self.load_config()

    def load_config(self) -> None:
        """
        Load and parse the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)

        self.paths = self._make_dict(config, "paths")
        self.selection = self._make_dict(config, "selection")
        self.scheduling = self._make_dict(config, "scheduling")
        self.training = self._make_dict(config, "training")
        self.testing = self._make_dict(config, "testing")

    @staticmethod
    def _make_dict(config: configparser.ConfigParser, section: str) -> Dict[str, Any]:
        """
        Convert a configuration section into a dictionary with auto-cast values.

        Args:
            config (configparser.ConfigParser): Parsed configuration object.
            section (str): Section name.

        Returns:
            Dict[str, Any]: Dictionary with parsed and type-cast values.
        """
        result: Dict[str, Any] = {}
        if section in config:
            for key, raw_value in config[section].items():
                result[key] = ConfigManager._auto_cast(raw_value)
        return result

    @staticmethod
    def _auto_cast(value: str) -> Any:
        """
        Automatically cast a string value to bool, int, or float if possible.

        Casting priority:
            1) bool ("true"/"false", case-insensitive)
            2) int
            3) float
            4) str (fallback)

        Args:
            value (str): Raw string value.

        Returns:
            Any: Cast value.
        """
        val_lower = value.lower()

        if val_lower == "true":
            return True
        if val_lower == "false":
            return False

        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

        if ConfigManager._is_float(value):
            return float(value)

        return value

    @staticmethod
    def _is_float(value: str) -> bool:
        """
        Check whether a string represents a float value (excluding integers).

        Args:
            value (str): Input string.

        Returns:
            bool: True if value can be parsed as float, False otherwise.
        """
        if value.count(".") != 1:
            return False

        left, right = value.split(".")
        if not right.isdigit():
            return False

        if left.isdigit():
            return True

        if left.startswith("-") and left[1:].isdigit():
            return True

        return False