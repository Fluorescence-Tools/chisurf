"""
Test script for the Count Rate Analysis CLI.

This script demonstrates how to use the CLI programmatically and can be used
to verify that the CLI is working correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the repository root to the Python path if needed
repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from click.testing import CliRunner

from chisurf.plugins.tttr.tttr_count_rate_analysis.cli import cli


def test_cli_help():
    """Test the CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    print("=== CLI Help ===")
    print(result.output)
    assert result.exit_code == 0
    assert "Count Rate Analysis CLI" in result.output


def test_list_setups_command():
    """Test the list-setups command."""
    # Find a detector setups file
    # This is just an example - adjust the path to a real setups file in your environment
    setups_file = os.path.join(repo_root, "chisurf", "settings", "detector_setups.json")

    if not os.path.exists(setups_file):
        print(f"Warning: Setups file not found at {setups_file}")
        print("Skipping list-setups test")
        return

    runner = CliRunner()
    result = runner.invoke(cli, ["list-setups", setups_file])
    print("\n=== List Setups ===")
    print(result.output)
    assert result.exit_code == 0


def test_analyze_command():
    """Test the analyze command with sample files."""
    # Find sample TTTR files
    # This is just an example - adjust the paths to real files in your environment
    sample_dir = os.path.join(repo_root, "test", "data")

    # Find TTTR files in the sample directory
    tttr_files = []
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith((".ptu", ".ht3", ".pt3", ".t3r")):
                tttr_files.append(os.path.join(sample_dir, file))

    if not tttr_files:
        print("Warning: No TTTR files found for testing")
        print("Skipping analyze test")
        return

    # Find a detector setups file
    setups_file = os.path.join(repo_root, "chisurf", "settings", "detector_setups.json")

    if not os.path.exists(setups_file):
        print(f"Warning: Setups file not found at {setups_file}")
        print("Skipping analyze test")
        return

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        output_file = tmp.name

    try:
        # Run the analyze command
        runner = CliRunner()
        args = (
            ["analyze"]
            + tttr_files
            + ["--setup-file", setups_file, "--output", output_file, "--verbose"]
        )
        print(f"\n=== Running analyze command with args: {args} ===")
        result = runner.invoke(cli, args)
        print(result.output)

        # Check if the output file was created and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Output file created successfully: {output_file}")
            with open(output_file) as f:
                print("\n=== Output file content ===")
                print(f.read())
        else:
            print(f"Warning: Output file not created or empty: {output_file}")

    finally:
        # Clean up the temporary file
        if os.path.exists(output_file):
            os.unlink(output_file)


if __name__ == "__main__":
    print("Testing Count Rate Analysis CLI...")
    test_cli_help()
    test_list_setups_command()
    test_analyze_command()
    print("\nAll tests completed.")
