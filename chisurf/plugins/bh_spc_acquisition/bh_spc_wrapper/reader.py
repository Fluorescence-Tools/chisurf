import re

class BeckerHicklSPCSetupReader:
    """
    Reader for Becker & Hickl SPC Setup (.set) Files (e.g., from SPC-830 TCSPC modules).
    Parses section blocks and hardware/software parameters (#PR, #SP).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.sections = {}
        self.parameters = {}
        self._read_file()
        self._parse_parameters()

    def _read_file(self):
        """Reads the .set file and splits it into named sections."""
        with open(self.filepath, "rb") as f:
            raw = f.read()

        try:
            text = raw.decode("ascii", errors="ignore")
        except UnicodeDecodeError:
            text = raw.decode("latin1")

        current_section = None
        section_lines = []

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("*") and not line.startswith("*END"):
                if current_section:
                    self.sections[current_section] = "\n".join(section_lines)
                current_section = line.strip("*").strip()
                section_lines = []
            elif line == "*END":
                if current_section:
                    self.sections[current_section] = "\n".join(section_lines)
                    current_section = None
                    section_lines = []
            else:
                if current_section:
                    section_lines.append(line)

    def _parse_parameters(self):
        """Extracts all #PR and #SP parameters from all sections."""
        for section, content in self.sections.items():
            for line in content.splitlines():
                line = line.strip()
                match = re.match(r'#(PR|SP)\s+\[([\w_]+),([A-Z]),(.*)\]', line)
                if match:
                    param_type = match.group(1)  # PR or SP
                    param_name = match.group(2)
                    param_data_type = match.group(3)
                    param_value = match.group(4).strip()
                    self.parameters[param_name] = {
                        "section": section,
                        "type": param_type,
                        "data_type": param_data_type,
                        "value": param_value
                    }

    def get_sections(self):
        """Returns a list of all section names."""
        return list(self.sections.keys())

    def get_section_content(self, section_name):
        """Returns the raw content of a specific section."""
        return self.sections.get(section_name, None)

    def get_parameters(self):
        """Returns all extracted parameters as a detailed dictionary."""
        return self.parameters

    def get_parameter(self, name):
        """Returns a single parameter dictionary by name, or None if not found."""
        return self.parameters.get(name, None)

    def to_dict(self):
        """Returns a clean dictionary: param name → value (string form)."""
        return {name: param["value"] for name, param in self.parameters.items()}

    def to_dataframe(self):
        """Returns all parameters as a Pandas DataFrame (if pandas is installed)."""
        try:
            import pandas as pd
            return pd.DataFrame.from_dict(self.parameters, orient='index')
        except ImportError:
            raise ImportError("pandas is required for DataFrame export. Please install it via 'pip install pandas'.")