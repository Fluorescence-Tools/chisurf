import re
import yaml
import ast
from pathlib import Path

from qtpy import QtCore


# ---------------------------------------------------------------------------
# Part 1 — YAML round‑trip: list preservation through YAML serialization
# ---------------------------------------------------------------------------

def _convert_lists_to_strings(data):
    if isinstance(data, dict):
        return {k: _convert_lists_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return ", ".join(str(item) for item in data)
    return data


def _convert_string_to_list(value_str):
    if (
        ('{' in value_str and '}' in value_str)
        or ('[' in value_str and ']' in value_str)
    ):
        try:
            prepared_str = '[' + value_str.replace("'", '"') + ']'
            return ast.literal_eval(prepared_str)
        except (SyntaxError, ValueError):
            pass
    items = [item.strip() for item in value_str.split(',')]
    converted_items = []
    for item in items:
        lower = item.lower()
        if lower == 'true':
            converted_items.append(True)
        elif lower == 'false':
            converted_items.append(False)
        elif lower in ('none', 'null'):
            converted_items.append(None)
        elif item.isdigit():
            converted_items.append(int(item))
        elif re.match(r'^-?\d+(\.\d+)?$', item):
            converted_items.append(float(item))
        else:
            converted_items.append(item)
    return converted_items


def _get_dict_from_item(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _get_dict_from_item(value)
        elif isinstance(value, str) and ',' in value:
            result[key] = _convert_string_to_list(value)
        else:
            result[key] = value
    return result


def test_yaml_round_trip_preserves_lists(tmp_path):
    test_data = {
        'simple_list': [1, 2, 3, 4, 5],
        'string_list': ['a', 'b', 'c', 'd'],
        'mixed_list': [1, 'string', 3.14, True, None],
        'nested_list': [[1, 2], [3, 4]],
        'complex': {
            'list_of_dicts': [
                {'name': 'item1', 'value': 1},
                {'name': 'item2', 'value': 2}
            ],
            'nested': {
                'deep_list': [10, 20, 30]
            }
        }
    }

    yaml_path = tmp_path / "test.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data, f, default_flow_style=False)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        loaded_data = yaml.safe_load(f)

    string_data = _convert_lists_to_strings(loaded_data)
    retrieved_data = _get_dict_from_item(string_data)

    retrieved_path = tmp_path / "test_retrieved.yaml"
    with open(retrieved_path, 'w', encoding='utf-8') as f:
        yaml.dump(retrieved_data, f, default_flow_style=False)

    _verify_lists_preserved(test_data, retrieved_data)


def _verify_lists_preserved(original, retrieved):
    success = 0
    failure = 0
    for key, value in original.items():
        if isinstance(value, dict):
            if key not in retrieved or not isinstance(retrieved[key], dict):
                failure += 1
                continue
            s, f = _verify_dict_lists(key, value, retrieved[key])
            success += s
            failure += f
        elif isinstance(value, list):
            assert key in retrieved, f"{key} missing in retrieved"
            assert isinstance(retrieved[key], list), f"{key} is not a list"
            assert len(value) == len(retrieved[key]), f"{key} length mismatch"
            success += 1
    assert failure == 0, f"{failure} verifications failed"


def _verify_dict_lists(parent_key, original_dict, retrieved_dict):
    success = 0
    failure = 0
    for key, value in original_dict.items():
        if isinstance(value, dict):
            s, f = _verify_dict_lists(f"{parent_key}.{key}", value, retrieved_dict[key])
            success += s
            failure += f
        elif isinstance(value, list):
            assert key in retrieved_dict, f"{parent_key}.{key} missing"
            assert isinstance(retrieved_dict[key], list), f"{parent_key}.{key} is not a list"
            assert len(value) == len(retrieved_dict[key]), f"{parent_key}.{key} length mismatch"
            success += 1
    return success, failure


# ---------------------------------------------------------------------------
# Part 2 — Mock SettingsTreeModel: simulate editor model round‑trip
# ---------------------------------------------------------------------------

class MockQStandardItem:
    def __init__(self, text=""):
        self.text_value = text
        self.data_value = None
        self.children = []
        self.editable_value = True
        self.tooltip_value = ""

    def text(self):
        return self.text_value

    def setText(self, text):
        self.text_value = text

    def setData(self, value, role):
        if role == QtCore.Qt.EditRole:
            self.data_value = value

    def data(self, role):
        if role == QtCore.Qt.EditRole:
            return self.data_value
        return self.text_value

    def setEditable(self, editable):
        self.editable_value = editable

    def setToolTip(self, tooltip):
        self.tooltip_value = tooltip

    def appendRow(self, row):
        self.children.append(row)

    def child(self, row, column):
        if row < len(self.children):
            return self.children[row][column]
        return None

    def rowCount(self):
        return len(self.children)

    def hasChildren(self):
        return len(self.children) > 0


class MockSettingsTreeModel:
    def __init__(self):
        self.root = MockQStandardItem()

    def invisibleRootItem(self):
        return self.root

    def load_settings(self, settings_dict):
        self.root = MockQStandardItem()
        self._populate_model(settings_dict)

    def _populate_model(self, settings_dict, parent=None, path=""):
        if parent is None:
            parent = self.root
        for key, value in sorted(settings_dict.items()):
            key_item = MockQStandardItem(key)
            key_item.setEditable(False)
            value_item = MockQStandardItem()
            value_item.setData(value, QtCore.Qt.EditRole)
            if isinstance(value, dict):
                pass
            elif isinstance(value, (list, tuple)):
                if not value:
                    value_item.setText("")
                elif any(isinstance(item, dict) for item in value):
                    value_item.setText("[complex list - edit with caution]")
                else:
                    items_str = []
                    for item in value:
                        if item is None:
                            items_str.append("None")
                        else:
                            items_str.append(str(item))
                    value_item.setText(", ".join(items_str))
            else:
                value_item.setText(str(value))
            row = [key_item, value_item]
            if isinstance(value, dict):
                parent.appendRow(row)
                self._populate_model(value, key_item, path + "." + key if path else key)
            else:
                parent.appendRow(row)

    def get_settings_dict(self):
        return self._get_dict_from_item(self.root)

    def _get_dict_from_item(self, item):
        result_dict = {}
        for row in range(item.rowCount()):
            key_item = item.child(row, 0)
            value_item = item.child(row, 1)
            key = key_item.text()
            if key_item.hasChildren():
                value = self._get_dict_from_item(key_item)
            else:
                value = value_item.data(QtCore.Qt.EditRole)
                if isinstance(value, str):
                    value_str = value
                    lower = value_str.lower()
                    if lower == "true":
                        value = True
                    elif lower == "false":
                        value = False
                    elif value_str.isdigit():
                        value = int(value_str)
                    elif re.match(r'^-?\d+(\.\d+)?$', value_str):
                        value = float(value_str)
            result_dict[key] = value
        return result_dict


def test_mock_model_round_trip_preserves_plugin_lists(qapp, tmp_path):
    test_data = {
        'plugins': {
            'disabled_models': [
                'Et-Model free',
                'Dye-diffusion'
            ],
            'disabled_plugins': [
                'TTTR:Splitter',
                'TTTR:Correlate',
                'TTTR:Generate Decay',
                'Tools:Bayesian FRET Analysis',
                'Single-Molecule:FIDA-2D',
                'Single-Molecule:FIDA'
            ],
            'hide_disabled_models': True,
            'hide_disabled_plugins': True,
            'icons_enabled': True,
            'plugin_order': {},
            'toolbar_plugins': [
                'Tools:Histogram-Microtime',
                'FCS:Correlator',
                'Single-Molecule:Burst-Selection',
                'Single-Molecule:Burst MLE Lifetime Analysis',
                'Tools:ndXplorer'
            ]
        }
    }

    yaml_path = tmp_path / "test.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data, f, default_flow_style=False)

    model = MockSettingsTreeModel()
    model.load_settings(test_data)
    retrieved_data = model.get_settings_dict()

    retrieved_path = tmp_path / "test_retrieved.yaml"
    with open(retrieved_path, 'w', encoding='utf-8') as f:
        yaml.dump(retrieved_data, f, default_flow_style=False)

    toolbar = retrieved_data['plugins']['toolbar_plugins']
    assert isinstance(toolbar, list), f"toolbar_plugins is {type(toolbar)}, expected list"
    assert len(toolbar) > 0, "toolbar_plugins should not be empty"
    assert "FCS:Correlator" in toolbar
