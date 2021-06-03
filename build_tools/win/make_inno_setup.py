import sys
import os
import glob
import jinja2
import pathlib


# append the relative location you want to import from
# import your module stored in '../common'
module_path = pathlib.Path("../../")
source_dir = pathlib.Path("../../dist/win/").absolute()
output_dir = pathlib.Path("../../dist/").absolute()
license_file = str((module_path / "LICENSE").absolute())
vc_runtime_path = "VC++ runtimes/"
vc_runtimes = [os.path.basename(f) for f in glob.glob(vc_runtime_path+"/*.exe")]

sys.path.append(str(module_path.absolute()))
path = pathlib.Path(module_path)
print(sys.path)

setup_file = module_path / "setup.py"
ns = dict()
with open(setup_file) as f:
    lines = f.readlines()
    # omit last line so that setup() is not called
    s = "".join(lines[:-2])
    code = compile(s, '<string>', 'exec')
    exec(code)

# the parameters come from the setup.py
parameters = {
    "AppId": __app_id__,
    "AppName": __name__,
    "AppVersion": __version__,
    "AppPublisher": __author__,
    "AppPublisherURL": __url__,
    "AppSupportURL": __url__,
    "AppUpdatesURL": __url__,
    "DefaultGroupName": __name__,
    "SourceDir": source_dir,
    "Output_dir": output_dir,
    "gui_entry_points": gui_scripts,
    "LicenseFile": license_file,
    "vc_runtime_path": vc_runtime_path,
    "vc_runtimes": vc_runtimes
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
