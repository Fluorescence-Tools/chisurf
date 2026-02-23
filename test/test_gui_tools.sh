#! /usr/bin/env bash

cd ..
### GUI tools
# broken
# working
python -m chisurf.plugins.traj.traj_align
python -m chisurf.plugins.traj.traj_convert
python -m chisurf.plugins.modelling.fps_json_editor
python -m chisurf.plugins.traj.fret_trajectory
python -m chisurf.plugins.traj.traj_join
python -m chisurf.plugins.traj.potential_energy
python -m chisurf.plugins.traj.traj_remove_clashes
python -m chisurf.plugins.traj.traj_save_topology
python -m chisurf.plugins.traj.traj_rotate_translate
python -m chisurf.plugins.misc.code_editor
python -m chisurf.plugins.misc.f_test
python -m chisurf.plugins.kappa2_dist

