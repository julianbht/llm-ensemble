"""Base registry for adapter registration and lookup."""

from typing import Dict, Callable, Type, Any, NamedTuple


class RegistryMetadata(NamedTuple):
    """Metadata about a registered adapter."""
    name: str
    adapter_class: Type
    description: str
    config: Dict[str, Any]


class BaseRegistry:
    """Base registry for adapter registration and lookup.

    Provides decorator-based registration and name-based lookup.
    """

    def __init__(self):
        self._registry: Dict[str, RegistryMetadata] = {}

    def register(
        self,
        name: str,
        description: str = "",
        **config
    ) -> Callable[[Type], Type]:
        """Decorator to register an adapter class.

        Args:
            name: Unique identifier for this adapter
            description: Human-readable description
            **config: Static configuration (e.g., template_path)

        Usage:
            @registry.register(
                name="thomas-simple",
                description="Thomas et al. simple prompt",
                template_path="thomas-simple.jinja"
            )
            class ThomasSimplePromptBuilder:
                pass
        """
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"Adapter '{name}' already registered in {self.__class__.__name__}"
                )

            self._registry[name] = RegistryMetadata(
                name=name,
                adapter_class=cls,
                description=description,
                config=config
            )
            return cls
        return decorator

    def get_metadata(self, name: str) -> RegistryMetadata:
        """Get metadata for a registered adapter.

        Args:
            name: Adapter name

        Returns:
            RegistryMetadata with adapter class and config

        Raises:
            ValueError: If adapter not found
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(
                f"Unknown adapter '{name}'. Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> Dict[str, str]:
        """Return all registered adapters with descriptions."""
        return {
            name: meta.description
            for name, meta in self._registry.items()
        }
