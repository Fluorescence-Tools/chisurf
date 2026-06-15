#!/usr/bin/env python3
"""Educational example of using the MFDB (Multiparameter Fluorescence Database)
and the Burst Selection plugin to register and process single-molecule data.

This script runs completely in-process (no ZMQ sockets, ports, or background servers).
"""

from pathlib import Path
import chisurf
from chisurf.core.mfdb import FluorescenceDatabase, BurstPipeline

# 1. Locate the test SPC file
chisurf_dir = Path(chisurf.__file__).resolve().parent
spc_file = (
    chisurf_dir
    / "plugins"
    / "burst"
    / "burst_selection"
    / "tests"
    / "data"
    / "bh_spc132_sm_dna"
    / "m000.spc"
)

print(f"Using SPC file: {spc_file}")

# 2. Connect to the database explicitly
# (Pass a file path to connect to another DB, e.g., FluorescenceDatabase("my_experiments.db"))
# COMMENT: Use connector, ie, give example with connector, where target can be a local DB or remote DB.
db = FluorescenceDatabase()
print(f"Connected to database at: {db.db_path}")

# 3. Initialize the pipeline with the database connection
pipeline = BurstPipeline(db)

# 4. Run the analysis (registers raw file, runs detection, and registers processed output)
print("Running burst selection analysis...")
# COMMENT USE DICT and pass paras as kwargs **
bur_file = pipeline.run(spc_file, min_photons=20, time_window=1e-3)
print(f"Analysis complete! Saved burst data: {bur_file}")

# 5. Query the database to retrieve the full provenance lineage (Round trip)
print("\nRetrieved workflow lineage from database:")
for edge in pipeline.get_lineage():
    print(
        f"  {edge['source_node_type']}:{edge['source_node_id']} --[{edge['relationship_type']}]--> {edge['target_node_type']}:{edge['target_node_id']}"
    )

print("\nDone!")
