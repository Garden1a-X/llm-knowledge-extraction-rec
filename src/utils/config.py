"""
Configuration loader utility.

Loads and validates configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """Configuration class for the project."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to the configuration YAML file.
                         If None, uses default configs/config.yaml
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variable placeholders
        config = self._replace_env_vars(config)

        return config

    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively replace environment variable references in config.

        References should be in format: ${ENV_VAR_NAME}
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check if string contains env var reference
            if config.startswith("${") and config.endswith("}"):
                env_var = config[2:-1]
                return os.getenv(env_var, config)
            return config
        else:
            return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys with dot notation, e.g., 'data.raw_dir'

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None

    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})

    @property
    def knowledge_extraction_config(self) -> Dict[str, Any]:
        """Get knowledge extraction configuration."""
        return self._config.get('knowledge_extraction', {})

    @property
    def clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration."""
        return self._config.get('clustering', {})

    @property
    def graph_config(self) -> Dict[str, Any]:
        """Get graph configuration."""
        return self._config.get('graph', {})

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('model', {})

    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})


# Global config instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to configuration file. If None and config not yet loaded,
                     uses default path.

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)

    return _global_config
