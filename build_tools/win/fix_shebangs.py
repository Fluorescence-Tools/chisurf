# This script is intended to fix shebangs in setuptools entry points after installation by
# InnoSetup.
#
# see: http://www.entropyreduction.al/python/distutils/2017/09/21/bundle-python-app-w-inno-setup.html
#
# e.g.
# [Run]
# "C:\Program Files\AwesomeApp\python.exe fix_shebangs.py awesomeapp"
#
# For each argument {arg}, it will search for a file
#   .\Scripts\{arg}-script.py
# and fix the shebang to point to the correct interpreter.

import os, os.path, sys, inspect, time

if os.name != 'nt':
    raise OSError('Fix shebangs only designed for Windows platform Python at present')

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
scriptsDir = os.path.join(currentDir, 'Scripts')
sys.path.insert(0, currentDir)
sys.path.insert(0, scriptsDir)

# Prefer to log to a file, but if that
try:
    log = open(os.path.join(currentDir, 'install.log'), 'w')
except PermissionError:
    class Log(object):
        def __init__(self):
            self.write = print


    log = Log()

for script in sys.argv[1:]:
    log.write('De-mangling script {}\n'.format(script))

    interp = 'python.exe'
    scriptFile = os.path.join(scriptsDir, script + '-script.py')
    if not os.path.isfile(scriptFile):
        # For some reason gui_scripts have the extension .pyw
        interp = 'pythonw.exe'
        scriptFile = os.path.join(scriptsDir, script + '-script.pyw')
        if not os.path.isfile(scriptFile):
            log.write('Script {} does not exist.\n'.format(script))
            continue

    with open(scriptFile, 'r') as sh:
        scriptLines = sh.readlines()

    new_shebang = '#!"{}"\n'.format(os.path.join(currentDir, interp))

    log.write("New shebang for {}: {}\n".format(script, new_shebang))
    scriptLines[0] = new_shebang
    # Writing here may require administrator access!
    with open(scriptFile, 'w') as sh:
        sh.writelines(scriptLines)

if log.write is print:
    print("Couldn't create install.log, so waiting...")
    time.sleep(10.0)
else:
    log.close()
