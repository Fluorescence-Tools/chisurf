import re
import yaml


def _convert_string_to_list(value_str):
    items = [item.strip() for item in value_str.split(',')]
    converted_items = []
    for item in items:
        if item.lower() == 'true':
            converted_items.append(True)
        elif item.lower() == 'false':
            converted_items.append(False)
        elif item.lower() == 'none' or item.lower() == 'null':
            converted_items.append(None)
        elif item.isdigit():
            converted_items.append(int(item))
        elif re.match(r'^-?\d+(\.\d+)?$', item):
            converted_items.append(float(item))
        else:
            converted_items.append(item)
    return converted_items


def test_comma_string_to_list_round_trip(tmp_path):
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
            'toolbar_plugins': [
                'Tools:Histogram-Microtime',
                'FCS:Correlator',
                'Single-Molecule:Burst-Selection',
                'Single-Molecule:Burst MLE Lifetime Analysis',
                'Tools:ndXplorer'
            ]
        },
        'mixed_types_list': [1, 'string', 3.14, True, None],
        'nested': {
            'fit_windows_size': [350, 350],
            'polarization_options': ['vm', 'vv', 'vh'],
            'rebin': [1, 1]
        }
    }

    yaml_path = tmp_path / "test.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data, f, default_flow_style=False)

    string_data = {}
    for key, value in test_data.items():
        if isinstance(value, list):
            string_data[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            string_data[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    string_data[key][sub_key] = ", ".join(str(item) for item in sub_value)
                else:
                    string_data[key][sub_key] = sub_value
        else:
            string_data[key] = value

    retrieved_data = {}
    for key, value in string_data.items():
        if isinstance(value, str) and ',' in value:
            retrieved_data[key] = _convert_string_to_list(value)
        elif isinstance(value, dict):
            retrieved_data[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str) and ',' in sub_value:
                    retrieved_data[key][sub_key] = _convert_string_to_list(sub_value)
                else:
                    retrieved_data[key][sub_key] = sub_value
        else:
            retrieved_data[key] = value

    retrieved_path = tmp_path / "test_retrieved.yaml"
    with open(retrieved_path, 'w', encoding='utf-8') as f:
        yaml.dump(retrieved_data, f, default_flow_style=False)

    _verify_lists_restored(test_data, retrieved_data)


def _verify_lists_restored(original, retrieved):
    for key, value in original.items():
        if isinstance(value, list):
            assert key in retrieved, f"{key} missing in retrieved"
            assert isinstance(retrieved[key], list), f"{key} is not a list"
            assert len(value) == len(retrieved[key]), f"{key} length mismatch"
        elif isinstance(value, dict):
            assert key in retrieved, f"{key} missing in retrieved"
            assert isinstance(retrieved[key], dict), f"{key} is not a dict"
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    assert sub_key in retrieved[key], f"{key}.{sub_key} missing"
                    assert isinstance(retrieved[key][sub_key], list), f"{key}.{sub_key} is not a list"
                    assert len(sub_value) == len(retrieved[key][sub_key]), f"{key}.{sub_key} length mismatch"
