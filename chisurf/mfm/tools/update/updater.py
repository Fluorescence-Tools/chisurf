

# https://hackthology.com/how-to-write-self-updating-python-programs-using-pip-and-git.html

from subprocess import check_call as run
from getopt import getopt, GetoptError
RELEASE = 'master' # default release
SRC_DIR = "$HOME/.src" # checkout directory
UPDATE_CMD = ( # base command
'pip install --src="%s" --upgrade -e ' 
'git://github.com/timtadh/swork.git@%s#egg=swork'
)

@command
def update(args):
    try:
        opts, args = getopt(args, 'sr:', ['sudo', 'src=', 'release=', 'commit='])
    except(GetoptError) as err:
        log(err)
        usage(error_codes['option'])

    sudo = False
    src_dir = SRC_DIR
    release = RELEASE
    commit = None
    for opt, arg in opts:
        if opt in ('-s', '--sudo'):
            sudo = True
        elif opt in ('-r', '--release'):
            release = arg
        elif opt in ('--src',):
            src_dir = arg
        elif opt in ('--commit',):
            commit = arg

    if release[0].isdigit(): ## Check if it is a version
        release = 'r' + release
    release = 'origin/' + release ## assume it is a branch

    if commit is not None: ## if a commit is supplied use that
        cmd = UPDATE_CMD % (src_dir, commit)
    else:
        cmd = UPDATE_CMD % (src_dir, release)

    if sudo:
        run('sudo %s' % cmd)
    else:
        run(cmd)

