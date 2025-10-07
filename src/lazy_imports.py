"""Lazy import utilities for memory optimization."""

import importlib.util
import sys
from typing import Any, Dict, Optional


class LazyImport:
    """Lazy import wrapper that only imports modules when first accessed."""

    def __init__(self, module_name: str, package: Optional[str] = None):
        """Initialize lazy import.

        Args:
            module_name: Name of the module to import
            package: Package name for relative imports
        """
        self._module_name = module_name
        self._package = package
        self._module: Optional[Any] = None
        self._imported = False

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the lazily imported module."""
        if not self._imported:
            self._import()
        return getattr(self._module, name)

    def _import(self) -> None:
        """Import the module."""
        try:
            if self._package:
                self._module = importlib.import_module(self._module_name, self._package)
            else:
                self._module = importlib.import_module(self._module_name)
            self._imported = True
        except ImportError as e:
            raise ImportError(f"Could not import {self._module_name}: {e}")

    def __repr__(self) -> str:
        """String representation."""
        if self._imported:
            return f"LazyImport({self._module_name}, imported=True)"
        return f"LazyImport({self._module_name}, imported=False)"


class LazyImports:
    """Registry for lazy imports to avoid duplicate imports."""

    _imports: Dict[str, LazyImport] = {}

    @classmethod
    def get(cls, module_name: str, package: Optional[str] = None) -> LazyImport:
        """Get or create a lazy import."""
        key = f"{package}.{module_name}" if package else module_name
        if key not in cls._imports:
            cls._imports[key] = LazyImport(module_name, package)
        return cls._imports[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the import cache."""
        cls._imports.clear()


# Common lazy imports for heavy libraries
numpy = LazyImports.get("numpy")
sentence_transformers = LazyImports.get("sentence_transformers")
transformers = LazyImports.get("transformers")
tokenizers = LazyImports.get("tokenizers")
tiktoken = LazyImports.get("tiktoken")
openai = LazyImports.get("openai")
cohere = LazyImports.get("cohere")
model2vec = LazyImports.get("model2vec")
torch = LazyImports.get("torch")
requests = LazyImports.get("requests")
