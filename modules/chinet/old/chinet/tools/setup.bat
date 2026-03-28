copy /Y tools\.condarc %userprofile%
copy /Y tools\conda_build_config.yaml conda_build_config.yaml
mamba install -y boa doxygen cmake
