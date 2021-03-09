#! /usr/bin/env bash

cd ..
### GUI tools
# broken
python -m chisurf.gui.tools.structure.fret_screening
# working
python -m chisurf.gui.tools.structure.align_trajectory
python -m chisurf.gui.tools.structure.convert_trajectory
python -m chisurf.gui.tools.structure.create_av_json
python -m chisurf.gui.tools.structure.fret_trajectory
python -m chisurf.gui.tools.structure.join_trajectories
python -m chisurf.gui.tools.structure.potential_energy
python -m chisurf.gui.tools.structure.remove_clashed_frames
python -m chisurf.gui.tools.structure.save_topology
python -m chisurf.gui.tools.structure.rotate_translate_trajectory
python -m chisurf.gui.tools.code_editor
python -m chisurf.gui.tools.configuration_editor
python -m chisurf.gui.tools.f_test
python -m chisurf.gui.tools.fret.calculator
python -m chisurf.gui.tools.kappa2_distribution

python -m chisurf.gui.tools.tttr.convert
python -m chisurf.gui.tools.tttr.correlate
python -m chisurf.gui.tools.tttr.decay
