import sys
import os
import pathlib
import re

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Create a simple test to verify the logic of the download-first approach
def test_download_logic():
    # Test remote URLs
    remote_urls = [
        "https://www.peulen.xyz/downloads/chisurf/conda/win-64/chisurf-25.06.23-py310h2b93aba_0.tar.bz2",
        "http://example.com/chisurf/latest.tar.bz2",
        "ftp://example.com/chisurf/latest.tar.bz2"
    ]
    
    # Test local paths
    local_paths = [
        "Q:\\chisurf\\conda\\win-64\\chisurf-25.06.23-py310h2b93aba_0.tar.bz2",
        "C:\\Users\\user\\Downloads\\chisurf-latest.tar.bz2",
        "\\\\server\\share\\chisurf-latest.tar.bz2"
    ]
    
    # Test the logic for remote URLs
    print("Testing remote URLs:")
    for url in remote_urls:
        is_remote = bool(re.match(r'^(https?|ftp)://', url))
        print(f"  {url} -> is_remote={is_remote}, expected=True")
        assert is_remote, f"Failed to detect remote URL: {url}"
    
    # Test the logic for local paths
    print("\nTesting local paths:")
    for path in local_paths:
        is_remote = bool(re.match(r'^(https?|ftp)://', path))
        print(f"  {path} -> is_remote={not is_remote}, expected=True")
        assert not is_remote, f"Incorrectly detected local path as remote URL: {path}"
    
    print("\nAll tests passed!")

# Run the test
if __name__ == "__main__":
    test_download_logic()