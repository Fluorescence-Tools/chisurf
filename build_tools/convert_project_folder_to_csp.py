from __future__ import annotations

import argparse
import pathlib

from chisurf.core.project.archive import (
    DATA_DIR,
    HISTORY_FILENAME,
    PROJECT_JSON,
    SESSION_FILENAME,
    ProjectArchive,
)


def convert_project_folder(input_path: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    """Convert a legacy project folder into a ``.csp`` archive."""
    input_path = input_path.resolve()
    if input_path.is_dir():
        project_json = input_path / PROJECT_JSON
        source_root = input_path
    elif input_path.name == PROJECT_JSON:
        project_json = input_path
        source_root = input_path.parent
    else:
        raise ValueError(f"Expected a project folder or {PROJECT_JSON}: {input_path}")

    if not project_json.is_file():
        raise FileNotFoundError(project_json)

    if output_path.suffix.lower() != ".csp":
        output_path = output_path.with_suffix(".csp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archive = ProjectArchive()
    archive.write_bytes(PROJECT_JSON, project_json.read_bytes())

    for filename in (HISTORY_FILENAME, SESSION_FILENAME, "fit.json"):
        path = source_root / filename
        if path.is_file():
            archive.write_bytes(filename, path.read_bytes())

    for path in source_root.iterdir():
        if path.name in {PROJECT_JSON, HISTORY_FILENAME, SESSION_FILENAME, "fit.json", DATA_DIR}:
            continue
        if path.is_file():
            archive.write_file(path.name, path)

    data_dir = source_root / DATA_DIR
    if data_dir.is_dir():
        for path in data_dir.rglob("*"):
            if not path.is_file():
                continue
            archive_name = f"{DATA_DIR}/{path.relative_to(data_dir).as_posix()}"
            archive.write_file(archive_name, path)

    return archive.save(output_path)


def main() -> int:
    """Run the project-folder to ``.csp`` converter."""
    parser = argparse.ArgumentParser(description="Convert a legacy ChiSurf project folder to .csp")
    parser.add_argument("project_folder", type=pathlib.Path)
    parser.add_argument("output_csp", type=pathlib.Path)
    args = parser.parse_args()

    output_path = convert_project_folder(args.project_folder, args.output_csp)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
