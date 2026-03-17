import sys
import os
import glob
import jinja2
import pathlib
import json
import tomli

script_dir = pathlib.Path(__file__).parent.absolute()
module_path = (script_dir / ".." / ".." / "chisurf").resolve()
setup_path = (script_dir / ".." / "..").resolve()
sys.path.append(str(module_path))
sys.path.append(str(setup_path))

import info


# Read all entry points (static + dynamic from plugins)
entry_points_json = setup_path / "rattler-recipe" / "entry_points.json"
if entry_points_json.exists():
    with open(entry_points_json, 'r') as f:
        ep_data = json.load(f)
        all_entry_points = ep_data.get("entry_points", [])
else:
    # Fallback: read from pyproject.toml only if entry_points.json doesn't exist
    print(f"WARNING: {entry_points_json} not found. Using pyproject.toml only.")
    pyproject_path = setup_path / "pyproject.toml"
    with open(pyproject_path, 'rb') as f:
        pyproject = tomli.load(f)
    scripts = pyproject.get("project", {}).get("scripts", {})
    gui_scripts = pyproject.get("project", {}).get("gui-scripts", {})
    all_entry_points = [f"{k} = {v}" for k, v in scripts.items()] + [f"{k} = {v}" for k, v in gui_scripts.items()]

# Extract GUI scripts from the full list for Inno Setup
# GUI scripts are those that map to GUI entry points (typically those in project.gui-scripts)
# For now, we'll parse them as name=module:func and determine which are GUI vs CLI
gui_scripts = {}
pyproject_path = setup_path / "pyproject.toml"
with open(pyproject_path, 'rb') as f:
    pyproject = tomli.load(f)
gui_scripts = pyproject.get("project", {}).get("gui-scripts", {})

# append the relative location you want to import from
# import your module stored in '../common'
source_dir = pathlib.Path("../../").resolve()
output_dir = pathlib.Path(os.environ.get("DIST_PATH", "../../dist")).resolve()
app_dir = pathlib.Path(os.environ.get("APP_PATH", "../../dist/win")).resolve()
license_file = str((source_dir / "LICENSE").resolve())
path = module_path
icon_file = str(path) + info.setup_icon

print("module_path:", module_path.resolve())
print("source_dir:", source_dir.resolve())
print("output_dir:", output_dir.resolve())
print("app_dir:", app_dir.resolve())
print("license_file:", license_file)
print("icon_file:", icon_file)

vc_runtime_path = "VC++ runtimes/"
vc_runtimes = [os.path.basename(f) for f in glob.glob(vc_runtime_path+"/*.exe")]


# the parameters come from the setup.py
parameters = {
    "AppId": info.__app_id__,
    "AppName": info.__name__,
    "AppVerName": f"{info.__name__} {info.__version__}" + (" (Dev)" if getattr(info, "__status__", "Dev") == "Dev" else ""),
    "AppVersion": info.__version__,
    "AppPublisher": info.__author__,
    "AppURL": info.__url__,
    "AppPublisherURL": info.__url__,
    "AppSupportURL": info.__url__,
    "AppUpdatesURL": info.__url__,
    "DefaultGroupName": info.__name__,
    "SourceDir": source_dir,
    "Output_dir": output_dir,
    "App_dir": app_dir,
    "LicenseFile": license_file,
    "vc_runtime_path": vc_runtime_path,
    "vc_runtimes": vc_runtimes,
    "SetupIconFile": icon_file,
    "gui_entry_points": gui_scripts,
    "IsDev": getattr(info, "__status__", "Dev") == "Dev",
}
inno_template = ""
with open('setup_template.jinja2', 'r') as fp:
    inno_template += fp.read()
t = jinja2.Template(inno_template)
inno_script = t.render(**parameters)

print("------ BEGIN INNO SETUP FILE ------")
print(inno_script)
print("------  END  INNO SETUP FILE ------")

with open('installer_config.iss', 'w') as fp:
    fp.write(inno_script)
