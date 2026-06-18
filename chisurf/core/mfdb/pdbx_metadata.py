"""Parse mmCIF dictionary files for vocabulary validation and metadata.

This module provides a comprehensive parser for mmCIF/PDBx dictionary files
in DDL2 format, with support for:
- Multi-line descriptions (semicolon-delimited blocks)
- Enumerated values (_item_enumeration)
- Data types (_item_type.code)
- Category and item metadata
- JSON caching for performance

The parsed dictionary data is used for:
- Vocabulary validation in sample creation
- GUI autocomplete for PDBx/flrCIF keys
- flrCIF export compliance checking
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DictItem:
    """A single item (field) from an mmCIF dictionary."""

    name: str
    category: str
    attribute: str
    description: str = ""
    type_code: str = ""
    mandatory: bool = False
    enumerations: List[str] = field(default_factory=list)
    enum_details: Dict[str, str] = field(default_factory=dict)
    parent: Optional[str] = None
    child: Optional[str] = None
    schema_table: str = ""
    schema_column: str = ""
    schema_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "attribute": self.attribute,
            "description": self.description,
            "type_code": self.type_code,
            "mandatory": self.mandatory,
            "enumerations": self.enumerations,
            "enum_details": self.enum_details,
            "parent": self.parent,
            "child": self.child,
            "schema_table": self.schema_table,
            "schema_column": self.schema_column,
            "schema_status": self.schema_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictItem":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            category=data.get("category", ""),
            attribute=data.get("attribute", ""),
            description=data.get("description", ""),
            type_code=data.get("type_code", ""),
            mandatory=data.get("mandatory", False),
            enumerations=data.get("enumerations", []),
            enum_details=data.get("enum_details", {}),
            parent=data.get("parent"),
            child=data.get("child"),
            schema_table=data.get("schema_table", ""),
            schema_column=data.get("schema_column", ""),
            schema_status=data.get("schema_status", ""),
        )


@dataclass
class DictCategory:
    """A category (table) from an mmCIF dictionary."""

    name: str
    description: str = ""
    mandatory: bool = False
    key_item: str = ""
    items: Dict[str, DictItem] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "mandatory": self.mandatory,
            "key_item": self.key_item,
            "items": {attr: item.to_dict() for attr, item in self.items.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictCategory":
        """Create from dictionary."""
        category = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            mandatory=data.get("mandatory", False),
            key_item=data.get("key_item", ""),
        )
        for attr, item_data in data.get("items", {}).items():
            category.items[attr] = DictItem.from_dict(item_data)
        return category


class MmcifDictionary:
    """Parsed mmCIF dictionary with fast lookup.

    This class parses .dic files in DDL2 format and provides fast access
    to categories, items, enumerations, and descriptions.
    """

    DATA_DIR = Path(__file__).resolve().parent / "data"
    CACHE_PATH = DATA_DIR / "_dictionary_cache.json"
    CACHE_VERSION = 2
    
    BUNDLED_DICTS = [
        "mmcif_ddl.dic",
        "mmcif_std.dic", 
        "mmcif_pdbx_v50.dic",
        "mmcif_pdbx_v5_next.dic",
        "mmcif_ma.dic",
        "mmcif_ihm_ext.dic",
        "mmcif_ihm_flr_ext.dic",
        "chisurf_flr_ext.dic",  # ChiSurf-specific extensions
    ]

    def __init__(self, *dic_paths: Path) -> None:
        self._categories: Dict[str, DictCategory] = {}
        self._items: Dict[str, DictItem] = {}
        self._file_timestamps: Dict[Path, float] = {}
        
        for path in dic_paths:
            if path.exists():
                self._parse_file(path)
                self._file_timestamps[path] = path.stat().st_mtime

    def _parse_file(self, path: Path) -> None:
        """Parse a single .dic file."""
        current_save: Optional[str] = None
        current_item: Optional[DictItem] = None
        in_loop = False
        loop_tags: List[str] = []
        loop_data: List[List[str]] = []
        pending_descriptions: List[str] = []
        category_description: str = ""
        category_mandatory: bool = False
        category_key: Optional[str] = None
        pending_description: Optional[str] = None
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                stripped = line.strip()
                
                if not stripped:
                    continue
                
                if stripped.startswith("save_"):
                    if in_loop and current_item and loop_tags:
                        self._process_loop_data(current_save, current_item, loop_tags, loop_data)
                    elif current_item and not in_loop:
                        self._register_item(current_save, current_item)
                    
                    save_name = stripped[5:].strip()
                    if not save_name.startswith("_"):
                        current_save = save_name
                        category_description = ""
                        category_mandatory = False
                        category_key = None
                        current_item = None
                    else:
                        current_save = save_name
                        current_item = None
                    
                    in_loop = False
                    loop_tags = []
                    loop_data = []
                    pending_descriptions = []
                    continue
                
                if stripped == "save_":
                    if in_loop and current_save and loop_tags:
                        self._process_loop_data(current_save, current_item, loop_tags, loop_data)
                    elif current_item and not in_loop:
                        self._register_item(current_save, current_item)
                    current_save = None
                    current_item = None
                    in_loop = False
                    loop_tags = []
                    loop_data = []
                    pending_descriptions = []
                    pending_description = None
                    continue
                
                if not current_save:
                    continue
                
                if not current_save.startswith("_"):
                    if stripped.startswith("_category_key.name"):
                        category_key = self._extract_value(stripped)
                    elif stripped.startswith("_category.mandatory_code"):
                        val = self._extract_value(stripped)
                        category_mandatory = val.lower() == "yes"
                    elif stripped.startswith("_category.description"):
                        val = self._extract_value(stripped)
                        category_description = val
                    elif stripped.startswith("loop_"):
                        in_loop = True
                        loop_tags = []
                        loop_data = []
                        continue
                    elif in_loop:
                        if stripped.startswith("_"):
                            loop_tags.append(stripped.strip())
                        else:
                            loop_data.append(stripped.split())
                        continue
                    continue
                
                if stripped.startswith("_item.name"):
                    if in_loop and current_item and loop_tags:
                        self._process_loop_data(current_save, current_item, loop_tags, loop_data)
                    elif current_item and not in_loop:
                        self._register_item(current_save, current_item)
                    
                    item_name = self._extract_value(stripped)
                    if not item_name:
                        continue
                    
                    parts = item_name[1:].split(".", 1)
                    if len(parts) == 2:
                        item_category, item_attr = parts
                    else:
                        item_category = ""
                        item_attr = item_name[1:]
                    
                    current_item = DictItem(
                        name=item_name,
                        category=item_category,
                        attribute=item_attr,
                        description=pending_description or "",
                    )
                    # Reset pending description and handle pending multi-line descriptions
                    pending_description = None
                    if pending_descriptions:
                        current_item.description = " ".join(pending_descriptions)
                        pending_descriptions = []
                    in_loop = False
                    loop_tags = []
                    loop_data = []
                    continue
                
                # Process description first, before checking for current_item
                # This handles the case where description comes before _item.name
                if stripped.startswith("_item_description.description"):
                    val = self._extract_value(stripped)
                    if val.startswith(";"):
                        pending_descriptions = [val[1:].strip()]
                    elif val == "":
                        # Multi-line description: value starts on next line with ; delimiter
                        # Set a flag to capture the next ;-block as the description
                        pending_descriptions = [""]
                    else:
                        # Store description for current item or as pending
                        if current_item is not None:
                            current_item.description = val
                        else:
                            # Store as pending description for next item
                            pending_description = val
                    continue  # Skip further processing for description lines
                
                if current_item is None:
                    # If we have a pending description but no current item yet,
                    # we'll apply it when the item is created
                    continue
                
                elif stripped.startswith(";"):
                    if stripped.endswith(";"):
                        pending_descriptions.append(stripped[1:-1].strip())
                        current_item.description = " ".join(pending_descriptions)
                        pending_descriptions = []
                    else:
                        pending_descriptions.append(stripped[1:].strip())
                elif stripped.startswith("_item_type.code"):
                    current_item.type_code = self._extract_value(stripped)
                elif stripped.startswith("_item.mandatory_code"):
                    val = self._extract_value(stripped)
                    current_item.mandatory = val.lower() == "yes"
                elif stripped.startswith("_chisurf_schema.table_name"):
                    current_item.schema_table = self._extract_value(stripped)
                elif stripped.startswith("_chisurf_schema.column_name"):
                    current_item.schema_column = self._extract_value(stripped)
                elif stripped.startswith("_chisurf_schema.status"):
                    current_item.schema_status = self._extract_value(stripped)
                elif stripped.startswith("loop_"):
                    in_loop = True
                    loop_tags = []
                    loop_data = []
                    continue
                elif in_loop:
                    if stripped.startswith("_"):
                        loop_tags.append(stripped.strip())
                    else:
                        loop_data.append(stripped.split())
                    continue
            
            if in_loop and current_item and loop_tags:
                self._process_loop_data(current_save, current_item, loop_tags, loop_data)
            elif current_item and not in_loop:
                self._register_item(current_save, current_item)
        
        # Only create categories for save blocks that don't start with "_"
        # (categories don't have leading underscore, items do)
        if current_save and not current_save.startswith("_"):
            if current_save not in self._categories:
                self._categories[current_save] = DictCategory(
                    name=current_save,
                    description=category_description,
                    mandatory=category_mandatory,
                    key_item=category_key or "",
                )
        
            if in_loop and current_item and loop_tags:
                self._process_loop_data(current_save, current_item, loop_tags, loop_data)
            elif current_item and not in_loop:
                self._register_item(current_save, current_item)
        
        for item_name, item in self._items.items():
            if item.category and item.category not in self._categories:
                self._categories[item.category] = DictCategory(name=item.category)
            if item.category and item.category in self._categories:
                self._categories[item.category].items[item.attribute] = item

    def _extract_value(self, line: str) -> str:
        """Extract the value from a dictionary line."""
        idx = line.find(" ")
        if idx < 0:
            return ""
        return line[idx:].strip().strip("'").strip('"')

    def _register_item(self, current_save: str, current_item: DictItem) -> None:
        """Register an item in the dictionary."""
        if not current_item or not current_item.name:
            return
        # Only set category from current_save if it's not already set
        # This handles the case where we already parsed _item.name and set category correctly
        if current_save and current_save.startswith("_") and not current_item.category:
            current_item.category = current_save[1:]

        existing = self._items.get(current_item.name)
        if existing is not None:
            self._items[current_item.name] = self._merge_item(existing, current_item)
            return

        self._items[current_item.name] = current_item

    @staticmethod
    def _merge_item(existing: DictItem, incoming: DictItem) -> DictItem:
        """Merge repeated item declarations from extension dictionaries."""
        for attr in (
            "category",
            "attribute",
            "description",
            "type_code",
            "parent",
            "child",
            "schema_table",
            "schema_column",
            "schema_status",
        ):
            value = getattr(incoming, attr)
            if value:
                setattr(existing, attr, value)

        existing.mandatory = existing.mandatory or incoming.mandatory
        for value in incoming.enumerations:
            if value not in existing.enumerations:
                existing.enumerations.append(value)
        existing.enum_details.update(incoming.enum_details)
        return existing

    def _process_loop_data(self, current_save: str, current_item: DictItem, 
                          loop_tags: List[str], loop_data: List[List[str]]) -> None:
        """Process loop_ data block."""
        if not current_item or not loop_tags or not loop_data:
            return
        
        enum_value_idx = None
        enum_detail_idx = None
        
        for i, tag in enumerate(loop_tags):
            if "_item_enumeration.value" in tag:
                enum_value_idx = i
            elif "_item_enumeration.detail" in tag:
                enum_detail_idx = i
        
        if enum_value_idx is not None:
            for row in loop_data:
                if enum_value_idx < len(row):
                    value = row[enum_value_idx].strip().strip("'").strip('"')
                    if value:
                        current_item.enumerations.append(value)
                        if enum_detail_idx is not None and enum_detail_idx < len(row):
                            detail = row[enum_detail_idx].strip().strip("'").strip('"')
                            current_item.enum_details[value] = detail
        
        self._register_item(current_save, current_item)

    @classmethod
    def load_bundled(cls) -> "MmcifDictionary":
        """Load all bundled dictionary files."""
        cached = cls._load_cache_if_valid()
        if cached is not None:
            return cached
        
        dic_paths = [cls.DATA_DIR / fname for fname in cls.BUNDLED_DICTS]
        dic = cls(*dic_paths)
        dic.save_cache()
        return dic

    @classmethod
    def _load_cache_if_valid(cls) -> Optional["MmcifDictionary"]:
        """Load from cache if valid."""
        if not cls.CACHE_PATH.exists():
            return None
        
        cache_mtime = cls.CACHE_PATH.stat().st_mtime
        for fname in cls.BUNDLED_DICTS:
            dic_path = cls.DATA_DIR / fname
            if dic_path.exists() and dic_path.stat().st_mtime > cache_mtime:
                return None
        
        try:
            with open(cls.CACHE_PATH, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            if cache_data.get("version") != cls.CACHE_VERSION:
                return None
            
            dic = cls()
            for cat_name, cat_data in cache_data.get("categories", {}).items():
                dic._categories[cat_name] = DictCategory.from_dict(cat_data)
            for item_name, item_data in cache_data.get("items", {}).items():
                dic._items[item_name] = DictItem.from_dict(item_data)
            for fname, mtime in cache_data.get("timestamps", {}).items():
                dic._file_timestamps[cls.DATA_DIR / fname] = mtime
            return dic
        except (json.JSONDecodeError, OSError):
            return None

    def save_cache(self, cache_path: Optional[Path] = None) -> None:
        """Save to JSON cache file.
        
        Parameters
        ----------
        cache_path : Path, optional
            Path to save the cache. If not provided, uses the default CACHE_PATH.
        """
        path = cache_path if cache_path is not None else self.CACHE_PATH
        cache_data = {
            "version": self.CACHE_VERSION,
            "categories": {name: cat.to_dict() for name, cat in self._categories.items()},
            "items": {name: item.to_dict() for name, item in self._items.items()},
            "timestamps": {str(k): v for k, v in self._file_timestamps.items()},
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except OSError:
            pass

    @classmethod
    def load_cache(cls, cache_path: Path) -> "MmcifDictionary":
        """Load dictionary from a JSON cache file.
        
        Parameters
        ----------
        cache_path : Path
            Path to the cache file to load.
            
        Returns
        -------
        MmcifDictionary
            A new MmcifDictionary instance loaded from the cache.
        """
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            dic = cls()
            # Restore categories
            for name, cat_data in cache_data.get("categories", {}).items():
                category = DictCategory.from_dict(cat_data)
                dic._categories[name] = category
            
            # Restore items
            for name, item_data in cache_data.get("items", {}).items():
                item = DictItem.from_dict(item_data)
                dic._items[name] = item
            
            # Restore timestamps
            for k, v in cache_data.get("timestamps", {}).items():
                dic._file_timestamps[Path(k)] = v
            
            return dic
        except (json.JSONDecodeError, OSError):
            raise ValueError(f"Failed to load cache from {cache_path}")

    def get_category(self, name: str) -> Optional[DictCategory]:
        return self._categories.get(name)

    def get_item(self, full_name: str) -> Optional[DictItem]:
        return self._items.get(full_name)

    def get_enumerations(self, full_name: str) -> List[str]:
        item = self.get_item(full_name)
        return item.enumerations if item else []

    def get_description(self, full_name: str) -> str:
        item = self.get_item(full_name)
        if item:
            return item.description
        cat = self.get_category(full_name)
        if cat:
            return cat.description
        return ""

    def search_items(self, query: str) -> List[DictItem]:
        query_lower = query.lower()
        return [item for item in self._items.values() 
                if (query_lower in item.name.lower() or 
                    query_lower in item.description.lower())]

    def categories(self) -> List[str]:
        return sorted(self._categories.keys())

    def flr_categories(self) -> List[str]:
        return sorted([n for n in self._categories.keys() if n.startswith("flr_")])

    def validate_value(self, full_name: str, value: str) -> Optional[str]:
        item = self.get_item(full_name)
        if item is None:
            return f"Unknown item: {full_name}"
        if item.enumerations and value not in item.enumerations:
            return (f"Invalid value '{value}' for {full_name}. "
                    f"Allowed: {', '.join(item.enumerations)}")
        return None

    def suggest_values(self, category: str, attribute: str, prefix: str = "") -> List[str]:
        full_name = f"_{category}.{attribute}"
        enums = self.get_enumerations(full_name)
        if enums:
            prefix_lower = prefix.lower()
            return [v for v in enums if prefix_lower in v.lower()]
        return []


def validate_flr_sample(fields: Dict[str, str], dic: Optional[MmcifDictionary] = None) -> List[str]:
    if dic is None:
        dic = MmcifDictionary.load_bundled()
    errors = []
    for full_name, value in fields.items():
        if not full_name.startswith("_"):
            full_name = f"_{full_name}"
        err = dic.validate_value(full_name, str(value))
        if err:
            errors.append(err)
    return errors


def validate_flr_value(category: str, attribute: str, value: str,
                       dic: Optional[MmcifDictionary] = None) -> Optional[str]:
    if dic is None:
        dic = MmcifDictionary.load_bundled()
    full_name = f"_{category}.{attribute}"
    return dic.validate_value(full_name, str(value))


def suggest_pdbx_keys(prefix: str = "") -> List[Tuple[str, str]]:
    dic = MmcifDictionary.load_bundled()
    prefix_lower = prefix.lower()
    results = []
    for item in dic._items.values():
        if prefix_lower in item.name.lower():
            results.append((item.name, item.description))
    for cat_name, cat in dic._categories.items():
        if prefix_lower in cat_name.lower():
            results.append((cat_name, cat.description))
    results.sort(key=lambda x: x[0])
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mmCIF dictionary introspection")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--flr-categories", action="store_true")
    parser.add_argument("--category", type=str)
    parser.add_argument("--enums", type=str)
    parser.add_argument("--search", type=str)
    parser.add_argument("--validate", nargs=2, metavar=("FIELD", "VALUE"))
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    dic = MmcifDictionary.load_bundled()
    
    if args.list_categories:
        for cat in dic.categories():
            print(f"  {cat}")
    elif args.flr_categories:
        for cat in dic.flr_categories():
            print(f"  {cat}")
    elif args.category:
        cat = dic.get_category(args.category)
        if cat:
            print(f"Category: {cat.name}")
            print(f"Description: {cat.description}")
            print(f"Items: {list(cat.items.keys())}")
        else:
            print(f"Not found: {args.category}")
    elif args.enums:
        enums = dic.get_enumerations(args.enums)
        for e in enums:
            print(f"  {e}")
    elif args.search:
        results = dic.search_items(args.search)
        print(f"Found {len(results)} items")
        for item in results[:50]:
            print(f"  {item.name}: {item.description[:60]}")
    elif args.validate:
        err = dic.validate_value(args.validate[0], args.validate[1])
        print(err or "VALID")
    elif args.stats:
        print(f"Categories: {len(dic.categories())}")
        print(f"flrCIF: {len(dic.flr_categories())}")
        print(f"Items: {len(dic._items)}")
    else:
        parser.print_help()
