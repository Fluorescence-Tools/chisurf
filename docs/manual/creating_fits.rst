Creating fits
-------------

.. image:: _images/image_rId17.png
  :align: center

In ChiSurf a :strong:`Fit` combines data with a model (:strong:`Fig.3`). Fits are stored internally in the list :strong:`chisurf.fits`. New fits are added to this list using the graphical user interface in three steps (:strong:`Fig.8`).

:strong:`Fig.8 Creating fits / analysis in the graphical user interface.` The top list displays imported data (Data list). The bottom list displays fits/analysis (Fit list).

In the first step, select a dataset in the data list by clicking on an item in the list (:strong:`Fig.8`, 1). In the second step, select a model for the selected dataset from the model selection dropdown menu (:strong:`Fig.8`, 2). In the third step, click on the :strong:`'+Analyis`' button next to the dropdown menu to create a new fit. In the first step, you can select multiple datasets of the same type by holding the Shift key while selecting data.

A new analysis can be created in the programming shell as follows:

Here the number of the dataset can be a list of integers. The integers refer to the index in the :strong:`chisurf.imported_datasets` list. The order of fit can be changed by editing the number of the dataset in the first column of the fit/analysis list (:strong:`Fig.4`). Note, the order of the fits can matter, as parameters and variables of analysis are accessed through the fit index and the parameter name. Fits can also be created for grouped data sets. Selecting a fit in the fit list will active the corresponding fit windows.
