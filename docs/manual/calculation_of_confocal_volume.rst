Calculation of confocal volume
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note:

Use the provided excel sheet ":emphasis:`FCCS_calibration.xlsx`" to retrieve the results semi-automatically.

Don't forget to update each time your fit results from the respective measurement day!

Please note the different units (µm, m etc.) and unit conversions!

To determine the confocal volume, we need three different values:

Diffusion coefficient of the freely diffusing standard dye, here A488,

These values can be found in the literature for most common fluorophores.

Caution! They are temperature and solvent dependent, i.e. if you measure at low temperature or in a more viscous environment, you have to correct for the viscosity of the solution.

:strong:`DA488 = 414 µm²/s` (@25°C in ddH2O 2

Diffusion time :strong:`tD` from our fit: :strong:`85.7 µs`

Shape factor :strong:`s` from our fit: :strong:`5.84`

Diffusion time and diffusion coefficient are related by the following relationship:

Inserting our fit results and :emphasis:`D` into this equation, we obtain :strong:`w0` :strong:`= 0.377 µm`

Based on the value from :strong:`w0`, we can determine the height :strong:`z0` of the confocal volume:

For our example, :strong:`z0` :strong:`= 2.2 µm`

Finally, the confocal volume is assumed to have in good approximation an elliptical shape from which the volume :strong:`Veff` can be obtained using the following formula:

Thus, our confocal volume in the green excitation range has a size of :strong:`Veff,green` :strong:`= 1.74 fL`.
