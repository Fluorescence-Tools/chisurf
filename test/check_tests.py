import os
import subprocess
import sys

def run_pytest_on_files():
    test_dir = 'e:/dev/chisurf/test'
    files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    results = []
    
    # Filter by what's ignored in pyproject.toml
    ignored = [
        'test_gui_',
        'test_widget_',
        'test_all_plugins.py'
    ]
    
    files = [f for f in files if not any(ig in f for ig in ignored)]
    
    print(f"Checking {len(files)} files...")
    
    for f in files:
        f_path = os.path.join(test_dir, f)
        # Use --collect-only first
        cmd = ['pixi', 'run', 'pytest', '--collect-only', f_path]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if 'error' in res.stdout.lower() or 'error' in res.stderr.lower() or res.returncode != 0:
                print(f"FAILED Collection: {f}")
                # print(res.stdout)
                # print(res.stderr)
                results.append((f, 'COLLECT_ERROR', res.stdout + res.stderr))
            else:
                # Actually run the test
                cmd = ['pixi', 'run', 'pytest', f_path, '-k', 'not gui and not widget and not window']
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    print(f"FAILED Run: {f}")
                    results.append((f, 'RUN_ERROR', res.stdout + res.stderr))
                else:
                    print(f"PASSED: {f}")
        except Exception as e:
            print(f"EXCEPTION: {f} - {e}")
            
    return results

if __name__ == "__main__":
    results = run_pytest_on_files()
    for r in results:
        print("="*60)
        print(f"Error in {r[0]} ({r[1]}):")
        # Just print the first few lines of the error
        lines = r[2].splitlines()
        for l in lines:
            if 'error' in l.lower() or 'traceback' in l.lower() or 'exception' in l.lower() or 'failed' in l.lower() or 'e  ' in l:
                print(l)
        print("="*60)
