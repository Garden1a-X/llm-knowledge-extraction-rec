"""
Configuration loader utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}

        # Load environment variables
        load_dotenv()

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Dictionary containing configuration
        """
        if config_name in self.configs:
            return self.configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.configs[config_name] = config
        return config

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files from the config directory.

        Returns:
            Dictionary mapping config names to their contents
        """
        all_configs = {}

        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            all_configs[config_name] = self.load_config(config_name)

        return all_configs

    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            config_name: Name of the configuration file
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Example:
            >>> config_loader.get('config', 'data.raw_data_path', './data/raw')
        """
        config = self.load_config(config_name)

        # Support nested key access with dot notation
        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_api_key(self, provider: str) -> str:
        """
        Get API key from environment variables.

        Args:
            provider: LLM provider name ('openai', 'anthropic', etc.)

        Returns:
            API key string

        Raises:
            ValueError: If API key is not found
        """
        llm_config = self.load_config('llm_config')

        provider_config = llm_config.get('llm', {}).get(provider, {})
        env_var_name = provider_config.get('api_key_env')

        if not env_var_name:
            raise ValueError(f"API key environment variable not configured for provider: {provider}")

        api_key = os.getenv(env_var_name)

        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {env_var_name}\n"
                f"Please set it in your .env file or environment."
            )

        return api_key


# Global config loader instance
_config_loader = None


def get_config_loader(config_dir: str = "configs") -> ConfigLoader:
    """
    Get or create a global ConfigLoader instance.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)

    return _config_loader
