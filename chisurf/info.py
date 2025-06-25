import pathlib
from datetime import date
today = date.today()

__name__ = "chisurf"
__author__ = "Thomas-Otavio Peulen"
__version__ = str(today.strftime("%y.%m.%d"))
__copyright__ = "Copyright (C) " + str(today.strftime('%y')) + " Thomas-Otavio Peulen"
__credits__ = ["Thomas-Otavio Peulen"]
__maintainer__ = "Thomas-Otavio Peulen"
__email__ = "thomas@peulen.xyz"
__url__ = "https://www.peulen.xyz/downloads/chisurf"
__license__ = 'GPL2.1'
__status__ = "Dev"
__description__ = """ChiSurf: an interactive global analysis platform for fluorescence data."""
__app_id__ = "{{ F25DCFFA-1234-4643-BC4F-2C3A20495937 }}"
LONG_DESCRIPTION = """ChiSurf: an interactive global analysis platform for fluorescence data."""
help_url = 'https://github.com/Fluorescence-Tools/chisurf/wiki'
update_url = 'https://github.com/Fluorescence-Tools/chisurf/releases'
setup_icon = "/gui/resources/icons/cs_logo.ico"
