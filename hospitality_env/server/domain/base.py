"""
Standalone base classes extracted from tau2-bench.

Replaces: tau2.utils.pydantic_utils.BaseModelNoExtra,
          tau2.environment.db.DB,
          tau2.environment.toolkit.ToolKitBase, ToolType, is_tool
"""

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, ConfigDict


# ============== BaseModelNoExtra ==============

class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============== DB ==============

class DB(BaseModelNoExtra):
    """Domain database base class."""

    @classmethod
    def load(cls, path) -> "DB":
        """Load from JSON file."""
        p = Path(path) if not isinstance(path, Path) else path
        with open(p, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def dump(self, path, exclude_defaults: bool = False, **kwargs: Any) -> None:
        """Dump to JSON file."""
        data = self.model_dump(exclude_defaults=exclude_defaults)
        p = Path(path) if not isinstance(path, Path) else path
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_hash(self) -> str:
        data = self.model_dump()
        return hashlib.md5(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get_statistics(self) -> dict[str, Any]:
        return {}


# ============== ToolType ==============

class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


# ============== is_tool decorator ==============

TOOL_ATTR = "__tool__"
TOOL_TYPE_ATTR = "__tool_type__"
MUTATES_STATE_ATTR = "__mutates_state__"
DISCOVERABLE_ATTR = "__discoverable__"


def is_tool(
    tool_type: ToolType = ToolType.READ,
    mutates_state: Optional[bool] = None,
):
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func):
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        setattr(func, MUTATES_STATE_ATTR, mutates_state)
        return func

    return decorator


# ============== ToolKitBase ==============

T = TypeVar("T", bound=DB)


class ToolKitType(type):
    """Metaclass that auto-discovers @is_tool methods."""

    def __init__(cls, name, bases, attrs):
        func_tools = {}
        for attr_name, method in attrs.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, TOOL_ATTR):
                func_tools[attr_name] = method

        @property
        def _func_tools(self) -> Dict[str, Callable]:
            all_func_tools = func_tools.copy()
            try:
                all_func_tools.update(super(cls, self)._func_tools)
            except AttributeError:
                pass
            return all_func_tools

        cls._func_tools = _func_tools


class ToolKitBase(metaclass=ToolKitType):
    """Base class for tool collections."""

    def __init__(self, db: Optional[T] = None):
        self.db: Optional[T] = db

    @property
    def tools(self) -> Dict[str, Callable]:
        return {name: getattr(self, name) for name in self._func_tools.keys()}

    def use_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name](**kwargs)

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def tool_type(self, tool_name: str) -> ToolType:
        return getattr(self.tools[tool_name], TOOL_TYPE_ATTR)

    def get_tool_names(self) -> list[str]:
        return list(self.tools.keys())
