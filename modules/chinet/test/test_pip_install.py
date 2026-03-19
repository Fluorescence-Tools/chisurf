import subprocess
import sys
import os

def test_pip_install():
    """Test if the module can be installed with pip in development mode."""
    print("Testing pip installation...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pip install in development mode
    cmd = [sys.executable, "-m", "pip", "install", "-e", current_dir]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Installation successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Installation failed!")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False

if __name__ == "__main__":
    success = test_pip_install()
    sys.exit(0 if success else 1)