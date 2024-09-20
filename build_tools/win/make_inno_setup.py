import sys
import os
import glob
import jinja2
import pathlib

module_path = pathlib.Path("../../chisurf").absolute().resolve()
setup_path = pathlib.Path("../..").absolute().resolve()
sys.path.append(str(module_path.resolve()))
sys.path.append(str(setup_path.resolve()))
path = pathlib.Path(module_path)

import info
from setup import gui_scripts

# append the relative location you want to import from
# import your module stored in '../common'
source_dir = pathlib.Path("../../").resolve()
output_dir = pathlib.Path("../../dist/").resolve()
license_file = str((source_dir / "LICENSE").resolve())
icon_file = str(path) + info.setup_icon

print("module_path:", module_path.resolve())
print("source_dir:", source_dir.resolve())
print("output_dir:", output_dir.resolve())
print("license_file:", license_file)
print("icon_file:", icon_file)

vc_runtime_path = "VC++ runtimes/"
vc_runtimes = [os.path.basename(f) for f in glob.glob(vc_runtime_path+"/*.exe")]


# the parameters come from the setup.py
parameters = {
    "AppId": info.__app_id__,
    "AppName": info.__name__,
    "AppVersion": info.__version__,
    "AppPublisher": info.__author__,
    "AppURL": info.__url__,
    "AppPublisherURL": info.__url__,
    "AppSupportURL": info.__url__,
    "AppUpdatesURL": info.__url__,
    "DefaultGroupName": info.__name__,
    "SourceDir": source_dir,
    "Output_dir": output_dir,
    "LicenseFile": license_file,
    "vc_runtime_path": vc_runtime_path,
    "vc_runtimes": vc_runtimes,
    "SetupIconFile": icon_file,
    "gui_entry_points": gui_scripts
}


inno_template = ""
with open('setup_template.iss', 'r') as fp:
    inno_template += fp.read()
t = jinja2.Template(inno_template)
inno_script = t.render(**parameters)

print("------ BEGIN INNO SETUP FILE ------")
print(inno_script)
print("------  END  INNO SETUP FILE ------")

with open('setup.iss', 'w') as fp:
    fp.write(inno_script)
