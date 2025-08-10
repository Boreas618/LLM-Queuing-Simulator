import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, Callable, Union


class ExtensionLoader:
    """
    Generic extension loader for discovering and loading plugins from directories.

    This class provides a unified interface for loading extensions, supporting:
    - Directory-based discovery
    - Class-based and function-based extensions
    - Flexible filtering criteria
    - Consistent error handling and logging
    """

    def __init__(self,
                 extension_dir: str,
                 base_classes: Optional[List[Type]] = None,
                 class_filter: Optional[Callable[[str, Type], bool]] = None,
                 function_filter: Optional[Callable[[str, Any], bool]] = None,
                 auto_instantiate: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the extension loader.

        Args:
            extension_dir: Directory to search for extensions
            base_classes: List of base classes that extensions should inherit from
            class_filter: Function to filter discovered classes (name, class) -> bool
            function_filter: Function to filter discovered functions (name, func) -> bool
            auto_instantiate: Whether to automatically instantiate discovered classes
            logger: Logger instance for reporting discovery results
        """
        self.extension_dir = extension_dir
        self.base_classes = base_classes or []
        self.class_filter = class_filter
        self.function_filter = function_filter
        self.auto_instantiate = auto_instantiate
        self.logger = logger or logging.getLogger(__name__)

    def load_extensions(self) -> Dict[str, Any]:
        """
        Load all extensions from the configured directory.

        Returns:
            Dictionary mapping extension names to their classes/functions/instances
        """
        extensions = {}
        plugin_path = Path(self.extension_dir)

        if not plugin_path.exists():
            self.logger.warning(
                f"Extension directory '{self.extension_dir}' does not exist")
            return extensions

        for file_path in plugin_path.glob("*.py"):
            # Skip files like __init__.py
            if file_path.name.startswith("_"):
                continue

            extension_name = file_path.stem
            module_name = f"{plugin_path.name}.{extension_name}"

            try:
                module = importlib.import_module(module_name)
                discovered = self._discover_in_module(module, extension_name)
                extensions.update(discovered)

            except ImportError as e:
                self.logger.error(
                    f"Could not import extension {file_path.name}: {e}")

        return extensions

    def _discover_in_module(self, module, extension_name: str) -> Dict[str, Any]:
        """
        Discover extensions within a loaded module.

        Args:
            module: The loaded module to search
            extension_name: Name of the extension file (for naming)

        Returns:
            Dictionary of discovered extensions
        """
        discovered = {}

        # Discover classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only consider classes defined in this module (not imported)
            if obj.__module__ != module.__name__:
                continue

            if self._is_valid_class(name, obj):
                key = self._get_extension_key(name, extension_name, obj)
                value = obj() if self.auto_instantiate else obj
                discovered[key] = value
                self.logger.info(f"Discovered class extension: '{key}'")

        # Discover functions (fallback for function-based extensions)
        if not discovered and self.function_filter:
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ != module.__name__:
                    continue

                if self.function_filter(name, obj):
                    key = self._get_extension_key(name, extension_name, obj)
                    discovered[key] = obj
                    self.logger.info(f"Discovered function extension: '{key}'")

        return discovered

    def _is_valid_class(self, name: str, obj: Type) -> bool:
        """Check if a class should be included as an extension."""
        # Skip the base classes themselves
        if self.base_classes and obj in self.base_classes:
            return False

        # Check inheritance requirements
        if self.base_classes and not any(issubclass(obj, base_class) for base_class in self.base_classes):
            return False

        # Apply custom filter
        if self.class_filter and not self.class_filter(name, obj):
            return False

        return True

    def _get_extension_key(self, name: str, extension_name: str, obj: Any) -> str:
        """
        Determine the key to use for an extension in the result dictionary.

        Args:
            name: The name of the class/function
            extension_name: The name of the extension file
            obj: The class/function object

        Returns:
            The key to use for this extension
        """
        # Default to using the file name as the key
        return extension_name


class PolicyLoader(ExtensionLoader):
    """Specialized loader for scheduling policies."""

    def __init__(self, policy_dir: str = 'policies', logger: Optional[logging.Logger] = None):
        from policies.base import Policy, LocalPolicy, GlobalPolicy

        super().__init__(
            extension_dir=policy_dir,
            base_classes=[Policy, LocalPolicy, GlobalPolicy],
            class_filter=self._policy_filter,
            auto_instantiate=False,
            logger=logger
        )

    def _policy_filter(self, name: str, obj: Type) -> bool:
        """Filter for policy classes."""
        from policies.base import Policy, LocalPolicy, GlobalPolicy

        # Must be a subclass of Policy but not one of the base classes
        return (issubclass(obj, Policy) and
                obj not in [Policy, LocalPolicy, GlobalPolicy])

    def _discover_in_module(self, module, extension_name: str) -> Dict[str, Any]:
        """
        Override discovery to handle multiple policy classes per file.
        For policies, we prefer Local policies over Global policies for naming consistency.
        """
        from policies.base import LocalPolicy

        discovered = {}
        valid_classes = []

        # Collect all valid classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if self._is_valid_class(name, obj):
                valid_classes.append((name, obj))

        if valid_classes:
            # Prefer LocalPolicy subclasses for consistent naming
            local_policies = [
                (name, obj) for name, obj in valid_classes if issubclass(obj, LocalPolicy)]
            if local_policies:
                name, obj = local_policies[0]
            else:
                name, obj = valid_classes[0]

            key = self._get_extension_key(name, extension_name, obj)
            discovered[key] = obj
            self.logger.info(f"Discovered class extension: '{key}'")

        return discovered


class ModelConfigLoader(ExtensionLoader):
    """Specialized loader for model configurations."""

    def __init__(self, model_dir: str = 'models', logger: Optional[logging.Logger] = None):
        super().__init__(
            extension_dir=model_dir,
            base_classes=[],  # Will check dynamically
            class_filter=self._model_config_filter,
            function_filter=self._model_function_filter,
            auto_instantiate=False,
            logger=logger
        )

    def _model_config_filter(self, name: str, obj: Type) -> bool:
        """Filter for model configuration classes."""
        return (name.endswith('Config') and
                not name.startswith('Base') and
                hasattr(obj, 'get_num_attention_heads'))

    def _model_function_filter(self, name: str, obj: Any) -> bool:
        """Filter for model configuration functions (fallback)."""
        # This could be used for backward compatibility with function-based configs
        return False  # Disable for now, focus on class-based configs

    def find_model_config(self, model_id: str) -> Optional[Any]:
        """
        Find a model configuration that matches the given model ID.

        Args:
            model_id: The model identifier to search for

        Returns:
            The matching model configuration class or None
        """
        extensions = self.load_extensions()

        # Look for exact matches first
        model_name_lower = model_id.lower()
        for key, config_class in extensions.items():
            if key in model_name_lower:
                self.logger.info(
                    f"Found model config: {key} for model: {model_id}")
                return config_class

        # No match found
        self.logger.warning(f"No model configuration found for: {model_id}")
        return None
