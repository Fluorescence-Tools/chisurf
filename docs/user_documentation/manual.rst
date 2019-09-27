*****
Manual
*****


Installation
============

Major releases
--------------

Major releases are distributed using pre-packed installation files for Windows
and MacOS. These installation files can be downloaded from the GitHub repository.

`The Python language <https://github.com/Fluorescence-Tools/ChiSurf/releases>`_

Conda releases
--------------

Alternatively the software can be installed using the Conda

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

distribution system from the conda channel `fluorescence-tools`.


Current release
---------------

The current release / the development version of the software can be installed
directly from the GitHub repository.

`Releases <https://github.com/Fluorescence-Tools/chisurf/releases>`_

It is recommended not to use development releases for productive purposes.


Introduction
============

General
-------

.. figure:: ../pictures/fig1_user_interface_overview.png
    :figwidth: 45%
    :align: right
    :figclass: align-left

    General overview: The different areas of the main program window


The main program window contains different areas:

1. Adding of data creation of new fits
2. Fitting and adjustment of fitting parameter of the currently selected fit
3. Change of plotting options of the current plot
4. Adding of a new dataset
5. List of loaded datasets
6. Adding of a new fit to the dataset currentls selected in 5)
7. Currently selected fit
8. Delete currently selected fit
9. Plotting area

By default three different docks (*Load*, *Analysis* and *Plot*) are displayed
on the left side of the window. By clicking on the names of the docks the user
can change the content displayed in the left dock. In (1) the user can add new
datasets and fits to the programm. In (2) the user can change the fitting
parameters of the fit currently selected in (7). In (3) the user can change
plotting options of the plot selected in (9) of the current fit (see Fig 1).



Experiment types
----------------

.. figure:: ../pictures/fig2_introduction_experiment_types.png
    :figwidth: 45%
    :align: right
    :figclass: align-left

    Selection of experiment types and setups


Before creation of a new fit the user has to choose the experiment type. Depending on the experiment type
and the experimental-setup associated to the experiment type the content within the group-box “Load data”
changes (compare Fig. 2).

Datasets
--------

.. figure:: ../pictures/intro_fig3_datasets.png
    :width: 45%
    :align: right
    :figclass: align-left

    Adding datasets


Before performing a fit or any data analysis the user has to add a dataset to the program.
First the user has to choose the experiment type and specify all relevant parameters
(details see sections below). As intermediate step

1. Adjust data-file parameters
2. Load-data
3. Optionally data-processing (correlation, creation of histograms)
4. Add-dataset

After addition of dataset at first nothing happens except that the dataset is now listed in the
dataset-table. By clicking on the row of a loaded dataset the currently selected dataset is changed.
The selected dataset is highlighted in green. If the user double clicks an element in the datasets
table the dataset and all related fits within the program are removed.

Fits
----

.. figure:: ../pictures/intro_fig4_fits.png
    :width: 350px
    :align: right
    :figclass: align-left

    Selection of datasets and adding fits

New fits can be added by the user to the currently selected dataset by clicking on *add fit*.
This creates a new fit containing the currently selected model and the currently selected dataset.
After addition of the fit the dataset and the fit are listed in (7).
The currently active fit is chosen by the user using the dropdown list (7). The content of the plotting area
(9), the fitting parameters (2) and the plotting-options area (3) depends on the currently selected fit.

.. figure:: ../pictures/intro_fig5_changing_active_fit.png
    :width: 45%
    :align: right
    :figclass: align-left

    Changing active fits

By selecting another fit out of the dropdown list the plot and fitting parameter are automatically
updated. In the Fitting pane all parameters of the currently selected fit are displayed. The fitting
range is usually determined automatically. It can be specified by unchecking the *autorange* checkbox.

Variables
---------

.. figure:: ../pictures/intro_fig6_variables.png
    :width: 45%
    :align: right
    :figclass: align-left

    Possible parameters of a fitting variable


All fitting variables have a name, are either fixed or not and can be linked to other fitting variables.
Fixing of fitting varibales is achived by checking the first checkbox. So far variables can only be linked
via a global-fit. If the fitting variable is linked the second checkbox is checked. The user can determine
the target of the link using the tooltip of the value field. Optionally Fitting variables have a lower or
upper limit. The lower and upper limit are either displayed or not. This is up to the developer of the
fitting model.


Plots
-----

.. figure:: ../pictures/intro_fig7_plots.png
    :width: 45%
    :align: center
    :figclass: align-left

    Plot area and plot option area

Each fitting model may contain different plot to represent the data and the fitting model. Depending on the selected
plot the content of the plot option area changes. The user may change the plot displayed in the plotting area
by changing the pane highlighted in red in Figure.

.. figure:: ../pictures/intro_fig8_plots_save.png
    :width: 45%
    :align: center
    :figclass: align-left

    Curves can be exported as CSV by right click on the curve and exporting the curve-data.

By default one dimensional data is plotted as blue line and the fitted curve is displayed as green line.
The colors and can be changed using the parameter window accessed using the right mouse button

.. figure:: ../pictures/intro_fig9_plots_surface_1.png
    :width: 45%
    :align: center
    :figclass: align-left

    Selection of plotted parameters


.. figure:: ../pictures/intro_fig9_plots_surface_2.png
    :width: 45%
    :align: center
    :figclass: align-left

    Inverting selections


.. figure:: ../pictures/intro_fig9_plots_surface_3.png
    :width: 45%
    :align: center
    :figclass: align-left

    Enabling and disabling selections


Error-estimation
================

Error estimation either by emcee or plain mcmc.
Short description

Support-Plane analysis
----------------------

