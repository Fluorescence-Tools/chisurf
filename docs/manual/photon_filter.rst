Photon filter
~~~~~~~~~~~~~

The photon filter in Step 2 allows for selecting photon from the photon stream. The selected photons that are correlated in a later step. Photons can be selected by the detection channel, micro time ranges, and by applying count rate filters to the photon stream. Macro time differences between photons, the selection mask, intensity time-traces, and micro time histograms of all photons and selected photons are displayed in the photon filtering window (:strong:`Fig.28`).

:emphasis:`Channel & micro time selection.` The channel selection widget (:strong:`Fig.28, 2a`) is used to define selections based on the detector number of registered photons and to define micro time ranges. By default, all photons in the photon stream are selected. Specifying channel numbers restricts the selected photons to the specified channels. Details and examples on the channel and micro time selection are summarized in :strong:`Tab.XX1`.

:emphasis:`Macro time interval.` The time between two consecutively registered photons can be used as a filter to select high count rate regions in a photon stream. Thresholds on minimum and maximum time between two consecutive photons are applied in the Macro time interval group (:strong:`Fig.28, 2b`). The selector in the macro time interval plot corresponds to the values set by the user in the "Macro time interval" group.

:emphasis:`Count rate filter.` For every photon in the stream photons within a define time window are selected. If for the selected photon and the defined time window more photons than are certain threshold (maximum number of photons) are found, the photon is not selected. The selection can be inverted to select high count rate regions.

Tab.1. Photon filter and plotting parameters.

The photon filter creates a folder for the intermediate steps of the analysis pipeline, e.g., the generated 'sl5' folder contains compressed JSON file with the filter parameters and the photon selection mask.
