"""
Configuration Loader Utility
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """Loads and manages configuration files."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config: {config_name}")
        self.configs[config_name] = config
        return config

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory.

        Returns:
            Dictionary mapping config_name to config_dict
        """
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            self.load_config(config_name)

        logger.info(f"Loaded {len(self.configs)} config files")
        return self.configs

    def get(self, config_name: str, key_path: str = None, default=None) -> Any:
        """
        Get a configuration value.

        Args:
            config_name: Name of the config file
            key_path: Dot-separated path to the value (e.g., "model.learning_rate")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if config_name not in self.configs:
            self.load_config(config_name)

        config = self.configs.get(config_name, {})

        if key_path is None:
            return config

        # Navigate nested dict using key_path
        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files.

        Args:
            *config_names: Names of config files to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}

        for config_name in config_names:
            if config_name not in self.configs:
                self.load_config(config_name)

            config = self.configs.get(config_name, {})
            merged.update(config)

        return merged
