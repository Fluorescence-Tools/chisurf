#!/usr/bin/env python

import argparse
import plistlib
import importlib
import sys
import pathlib
import os
import stat
import shutil
import jinja2

# append the local path and one above to be able to
# able to work with the modules in the current directory
# (and one above).
sys.path.append(
    os.getcwd()
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Update Apple Info.plist file with python module information.'
    )

    parser.add_argument(
        '-m',
        '--module',
        metavar='template',
        type=str,
        required=True,
        help='Inspected python module'
    )

    parser.add_argument(
        '-t',
        '--template',
        metavar='template',
        type=str,
        required=True,
        help='Template launcher file.'
    )

    parser.add_argument(
        '-p',
        '--plist_template',
        type=str,
        required=True,
        help='Template .plist file.'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output .plist file.'
    )

    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        default=None,
        help='Name that is written to the plist file as a version number.'
    )

    parser.add_argument(
        '-n',
        '--name',
        action='store_true',
        default=None,
        help='Display name of the software.'
    )

    parser.add_argument(
        '-e',
        '--executable',
        required=True,
        help='Name of the executable (place your executable into Contents/MacOS'
    )

    parser.add_argument(
        '-s',
        '--system_minimum',
        action='store_true',
        default="10.11.0",
        help='Minimum osx version.'
    )

    parser.add_argument(
        '-i',
        '--icon',
        type=str,
        default=None,
        help='The .icns file.'
    )

    parser.add_argument(
        '-c',
        '--copyright',
        type=str,
        default=None,
        help='Copyright string.'
    )

    args = parser.parse_args()

    template_file = args.template
    template_bytes = open(
        file=args.plist_template, mode='rb'
    ).read()

    module_string = args.module
    module = importlib.import_module(module_string)

    if args.version is None:
        print("No version number provided using string from module")
        version = module.__version__
    else:
        version = args.version

    if args.system_minimum is None:
        print("No minimum system version provided using 10.11.0.")
        system_minimum = "10.9.0"
    else:
        system_minimum = args.system_minimum

    if args.copyright is None:
        print("No copyright provided using module default.")
        try:
            try:
                copyright_info = module.copyright()
            except AttributeError:
                copyright_info = module.__copyright__
        except AttributeError:
            print("WARNING: module has no copyright information")
            copyright_info = ""
    else:
        copyright_info = args.copyright

    executable = args.executable
    plist_output_file = args.output

    if args.name is None:
        print("WARNING: no display name provided using executable name.")
        display_name = executable
    else:
        display_name = args.name

    print("Create an Info.plist file")
    print("=========================")
    print("Module: %s" % args.module)
    print("Template file: %s" % template_file)
    print("Output file: %s" % plist_output_file)
    print("Executable name: %s" % args.executable)
    print("Minimum system version: %s" % system_minimum)
    print("Copyright string: %s" % copyright_info)
    print("Display name: %s" % display_name)
    print("Software_version: %s" % version)

    target = dict()
    d = plistlib.loads(template_bytes)
    target.update(d)

    plist_path = pathlib.Path(plist_output_file).parent
    icns_file_path = ""
    if isinstance(args.icon, str):
        shutil.copy(args.icon, plist_path)
        icns_file_path = os.path.basename(args.icon)

    info = {
        'CFBundleDisplayName': display_name,
        'CFBundleExecutable': executable,
        'CFBundleName': display_name,
        'CFBundleShortVersionString': version,
        'CFBundleVersion': version,
        'CFBundleInfoDictionaryVersion': version,
        'LSMinimumSystemVersion': system_minimum,
        'NSHumanReadableCopyright': copyright_info,
        'CFBundleIconFile': icns_file_path
    }
    target.update(info)

    with open(plist_output_file, 'wb') as fp:
        plistlib.dump(
            value=target,
            fp=fp
        )
    launch_template = ""
    with open(template_file, 'r') as fp:
        launch_template += fp.read()
    t = jinja2.Template(
        launch_template
    )
    launch_script = t.render(app_name=module_string)
    executable_file = plist_path / "MacOS" / executable
    print("LAUNCH FILE")
    print("-----------")
    print(launch_script)
    with open(executable_file, 'w') as fp:
        fp.write(launch_script)

    st = os.stat(str(executable_file))
    os.chmod(
        str(executable_file), st.st_mode | stat.S_IEXEC
    )

