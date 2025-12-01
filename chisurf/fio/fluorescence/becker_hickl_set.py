import re
from pathlib import Path


class BeckerHicklSetReader:
    """
    Reader for Becker & Hickl SPC .set files.

    Attributes
    ----------
    params : dict
        Dictionary of raw SPC parameters, e.g. {'SP_SYN_FQ': -50.98, 'SP_TAC_TC': 1.83e-11, ...}

    Properties
    ----------
    macro_time_resolution : float
        Sync period in seconds (coarse time bin width).
    micro_time_resolution : float
        TAC conversion factor in seconds (micro-time channel width).
    tac_range : float
        Full TAC span in seconds.
    """

    _LINE_REGEX = re.compile(
        r"SP_([A-Z0-9_]+),"  # parameter name
        r"([FI]),"           # type: F=float, I=int
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\]"  # value
    )

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath : str or Path
            Path to the .set file.
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.params = {}
        self._read()

    def _read(self):
        """Internal: read file, decode, and parse SP_ parameters."""
        raw = self.filepath.read_bytes()
        text = raw.decode('latin1', errors='ignore')
        for name, t, val in self._LINE_REGEX.findall(text):
            if t == 'F':
                self.params[name] = float(val)
            else:  # 'I'
                self.params[name] = int(val)

    def get_param(self, name, default=None):
        """
        Get raw SPC parameter.

        Parameters
        ----------
        name : str
            Parameter name without the "SP_" prefix (e.g. "SYN_FQ").
        default : any
            Value to return if not found.

        Returns
        -------
        float or int or default
        """
        return self.params.get(name, default)

    @property
    def macro_time_resolution(self):
        """
        Coarse (macro) time resolution in seconds:
          Δt_macro = 1 / |SP_SYN_FQ| (SP_SYN_FQ in MHz)
        """
        fq = self.get_param('SYN_FQ')
        if fq is None or fq == 0:
            return None
        return 1.0 / (abs(fq) * 1e6)

    @property
    def micro_time_resolution(self):
        """
        Fine (micro) time resolution in seconds:
          = SP_TAC_TC
        """
        return self.get_param('TAC_TC')

    @property
    def tac_range(self):
        """
        Full TAC span in seconds:
          = SP_TAC_R
        """
        return self.get_param('TAC_R')

    def summary(self):
        """Print a quick overview of the key timing parameters."""
        print(f"File: {self.filepath}")
        print(f"  SP_SYN_FQ (MHz):          {self.get_param('SYN_FQ')}")
        print(f"  Macro-time resolution:    {self.macro_time_resolution * 1e9:.3f} ns")
        print(f"  SP_TAC_TC (s):            {self.micro_time_resolution:.3e} s")
        print(f"  Micro-time resolution:    {self.micro_time_resolution * 1e12:.3f} ps")
        print(f"  SP_TAC_R (s):             {self.tac_range:.3e} s")
        print(f"  TAC full-scale range:     {self.tac_range * 1e9:.3f} ns")


if __name__ == "__main__":
    # Example usage:
    reader = BeckerHicklSetReader("m_003.set")
    reader.summary()