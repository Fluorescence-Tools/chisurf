#!/usr/bin/env bash

# save the current directory and use this variable
# this variable is used at the end of the script
# to return to the original script directoy
export SCRIPT_DIR="."
SCRIPT_DIR=$("pwd")

echo "Creating an environemnt that will build the docs..."
conda env create -f ../../environment.yml --force -n build_doc
conda activate build_doc
echo "Installing the conda packages necessary to compile the documentation..."
conda install -y sphinx_bootstrap_theme sphinx sphinx_rtd_theme numpydoc sphinx-autodoc-typehints matplotlib=2.2.3
cd ../../docs

make clean
make html
make latex

cd _latex
make
conda deactivate
conda remove -y --name build_doc --all
cd $SCRIPT_DIR
