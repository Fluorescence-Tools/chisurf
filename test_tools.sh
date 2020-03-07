#! /usr/bin/env bash

# broken
python -m chisurf.tools.structure.fret_screening

# working
python -m chisurf.tools.structure.align_trajectory
python -m chisurf.tools.structure.convert_trajectory
python -m chisurf.tools.structure.create_av_json
python -m chisurf.tools.structure.fret_trajectory
python -m chisurf.tools.structure.join_trajectories
python -m chisurf.tools.structure.potential_energy
python -m chisurf.tools.structure.remove_clashed_frames
python -m chisurf.tools.structure.save_topology
python -m chisurf.tools.structure.rotate_translate_trajectory
python -m chisurf.tools.structure.save_topology
python -m chisurf.tools.code_editor
python -m chisurf.tools.configuration_editor
python -m chisurf.tools.f_test
python -m chisurf.tools.fret.calculator
python -m chisurf.tools.kappa2_distribution

python -m chisurf.tools.tttr.clsm_pixel_select
python -m chisurf.tools.tttr.convert
python -m chisurf.tools.tttr.correlate
python -m chisurf.tools.tttr.decay

