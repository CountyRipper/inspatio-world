"""CUT3R-surfel indexed KV memory research prototype.

All hooks are opt-in. Importing this package does not alter the upstream
InSpatio-World inference path.
"""

from .memory_context import ActiveLayerMemory, MemoryContext

__all__ = ["ActiveLayerMemory", "MemoryContext"]
