FRET Lines
"""""""""""""""""""""""""""""""""""""""""

.. _fret_lines:

Understanding FRET Lines
=======================

FRET lines are a fundamental tool in the analysis of Förster Resonance Energy Transfer (FRET) 
experiments. They provide a visual and quantitative representation of the relationship between 
two key lifetime parameters: the **fluorescence-averaged lifetime** (:math:`\tau_F`) and the 
**species-averaged lifetime** (:math:`\tau_X`).

What is a FRET Line?
-------------------

A FRET line represents the functional relationship:

.. math::
   \tau_X = f(\tau_F)

where:

- :math:`\tau_F = \frac{\langle \tau^2 \rangle}{\langle \tau \rangle}` (fluorescence-averaged lifetime)
- :math:`\tau_X = \langle \tau \rangle` (species-averaged lifetime)

For an ideal single-exponential donor fluorophore **without FRET**, :math:`\tau_F = \tau_X`, 
and the relationship follows a 1:1 line. However, when FRET is present, this relationship 
deviates from the 1:1 line in a characteristic manner that depends on the donor-acceptor 
distance distribution and the FRET efficiency.

Types of FRET Lines
------------------

ChiSurf provides two main types of FRET lines:

1. **Static FRET Line**
2. **Dynamic FRET Line**

Static FRET Line
~~~~~~~~~~~~~~~~

A **static FRET line** represents a system with a **single conformational state** where the 
donor-acceptor distance follows a Gaussian distribution. The line shows how :math:`\tau_F` and 
:math:`\tau_X` vary as the **mean distance** :math:`R` between donor and acceptor changes.

**Key characteristics:**

- Single Gaussian distance distribution with mean :math:`R` and width :math:`\sigma`
- Distance :math:`R` is varied along the line
- The line bends away from the 1:1 line as FRET efficiency increases (shorter distances)
- At large distances (no FRET), the line approaches the 1:1 line

**Mathematical basis:**

For a Gaussian distance distribution with mean :math:`R` and standard deviation :math:`\sigma`,
the FRET efficiency :math:`E` is given by:

.. math::
   E(R) = \int_0^\infty P(r) \cdot \frac{R_0^6}{R_0^6 + r^6} dr

where :math:`P(r)` is the Gaussian probability distribution and :math:`R_0` is the Förster radius.

The lifetimes are then calculated as:

.. math::
   \tau_X = \tau_0 \cdot (1 - E)
   \tau_F = \frac{\tau_X^2}{\tau_0}

where :math:`\tau_0` is the donor lifetime in the absence of FRET.

Dynamic FRET Line
~~~~~~~~~~~~~~~~~~

A **dynamic FRET line** represents a system with **two (or more) exchanging conformational states**, 
each with its own Gaussian distance distribution. The line shows how :math:`\tau_F` and :math:`\tau_X` 
vary as the **population fraction** of the states changes.

**Key characteristics:**

- Two (or more) Gaussian distance distributions with different mean distances
- The species fraction :math:`x_i` of each state is varied along the line
- The line shows a nonlinear transition between the limiting states
- At :math:`x_2 = 0` (pure state 1), the lifetimes correspond to state 1
- At :math:`x_2 = 1` (pure state 2), the lifetimes correspond to state 2

**Mathematical basis:**

For a two-state system with fractions :math:`x_1` and :math:`x_2 = 1 - x_1`, the observed lifetimes are:

.. math::
   \tau_X = x_1 \cdot \tau_{X,1} + x_2 \cdot \tau_{X,2}
   \tau_F = \frac{x_1 \cdot \tau_{X,1}^2 + x_2 \cdot \tau_{X,2}^2}{x_1 \cdot \tau_{X,1} + x_2 \cdot \tau_{X,2}}

where :math:`\tau_{X,i}` is the species-averaged lifetime of state :math:`i`.

Key Parameters
-------------

Both static and dynamic FRET lines depend on several fundamental parameters:

1. **Förster Radius** (:math:`R_0`)
   
   - Distance at which the FRET efficiency is 50%
   - Typical values: 30-60 Å for common dye pairs (e.g., Alexa 488/Alexa 647: ~52 Å)
   - Depends on the spectral overlap between donor emission and acceptor absorption
   - In ChiSurf: Set via ``R0`` parameter (default: 52 Å)

2. **Donor Lifetime Without FRET** (:math:`\tau_0`)
   
   - Fluorescence lifetime of the donor in the absence of FRET
   - Typical values: 1-10 ns
   - In ChiSurf: Set via ``t0`` parameter (default: 4 ns)

3. **Distance Distribution Width** (:math:`\sigma`)
   
   - Width of the Gaussian distance distribution
   - Accounts for conformational flexibility and dye linker flexibility
   - Typical values: 5-15 Å
   - In ChiSurf: Set via ``sigma`` parameter (default: 6 Å for examples)

4. **Orientation Factor** (:math:`\kappa^2`)
   
   - Describes the relative orientation of donor and acceptor dipoles
   - For isotropic averaging: :math:`\kappa^2 = 2/3 \approx 0.667`
   - In ChiSurf: Set via ``k2`` parameter

Creating FRET Lines in ChiSurf
------------------------------

Static FRET Line
~~~~~~~~~~~~~~~~

To create a static FRET line programmatically:

.. code-block:: python

   import numpy as np
   import chisurf.core.models.tcspc.fret
   import chisurf.core.fluorescence.fret.fret_line

   # Set up the R_DA axis
   chisurf.core.models.tcspc.fret.rda_axis = np.logspace(
       start=np.log(1),
       stop=np.log(500)
   )

   # Create a static FRET line with R0=52, sigma=6, tau0=4
   static_fl = chisurf.core.fluorescence.fret.fret_line.StaticFRETLine(
       n_points=100,
       parameter_range=(10, 100)  # Distance range in Angstrom
   )
   
   # Set the distribution width
   static_fl.sigma = 6.0
   
   # Calculate the FRET line
   static_fl.update()
   
   # Access the conversion function data
   tau_f, tau_x = static_fl.conversion_function
   
   # Get the polynomial approximation
   coeffs = static_fl.polynom_coefficients
   
   # Get string representation for plotting (e.g., in Origin)
   print(static_fl.conversion_function_string)

Dynamic FRET Line
~~~~~~~~~~~~~~~~~~

To create a dynamic FRET line with two states:

.. code-block:: python

   import numpy as np
   import chisurf.core.models.tcspc.fret
   import chisurf.core.fluorescence.fret.fret_line

   # Set up the R_DA axis
   chisurf.core.models.tcspc.fret.rda_axis = np.logspace(
       start=np.log(1),
       stop=np.log(500)
   )

   # Create a dynamic FRET line with two states
   # State 1: R1=40 Å, sigma1=6 Å
   # State 2: R2=80 Å, sigma2=6 Å
   dynamic_fl = chisurf.core.fluorescence.fret.fret_line.DynamicFRETLine(
       distance_1=40.0,
       distance_2=80.0,
       sigma_1=6.0,
       sigma_2=6.0,
       n_points=100,
       parameter_range=(0, 1)  # Vary x(G,2) from 0 to 1
   )
   
   # Calculate the FRET line
   dynamic_fl.update()
   
   # Access the conversion function data
   tau_f, tau_x = dynamic_fl.conversion_function
   
   # The parameter values represent x(G,2), the fraction of state 2
   x_values = dynamic_fl.parameter_values

FRET Line Properties
--------------------

Both static and dynamic FRET line objects provide the following properties and methods:

**Properties:**

- ``conversion_function``: Tuple of (tau_F, tau_X) arrays
- ``conversion_function_string``: Polynomial string representation for plotting
- ``transfer_efficency_string``: String representation of transfer efficiency
- ``fdfa_string``: String for FD/FA ratio
- ``fret_efficiencies``: Array of FRET efficiencies along the line
- ``fluorescence_averaged_lifetimes``: Array of tau_F values
- ``species_averaged_lifetimes``: Array of tau_X values
- ``parameter_values``: Array of the varied parameter (R or x)
- ``polynom_coefficients``: Polynomial coefficients approximating the conversion function

**For StaticFRETLine:**

- ``sigma``: Width of the Gaussian distance distribution
- ``mean_distance``: Mean distance R(G,1)

**For DynamicFRETLine:**

- ``mean_distance_1``: Mean distance of state 1 (R(G,1))
- ``mean_distance_2``: Mean distance of state 2 (R(G,2))
- ``sigma_1``: Width of state 1 distribution (s(G,1))
- ``sigma_2``: Width of state 2 distribution (s(G,2))

Visualizing FRET Lines
----------------------

FRET lines can be visualized using matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot static FRET line
   fig, ax = plt.subplots(figsize=(8, 6))
   tau_f, tau_x = static_fl.conversion_function
   ax.plot(tau_f, tau_x, 'b-', linewidth=2, label='Static FRET Line')
   ax.plot(tau_f, tau_f, 'r--', linewidth=1, label='1:1 line')
   ax.set_xlabel(r'$\tau_F$ (Fluorescence-avg lifetime) [ns]')
   ax.set_ylabel(r'$\tau_X$ (Species-avg lifetime) [ns]')
   ax.set_title('Static FRET Line\n$R_0$=52Å, $\sigma$=6Å, $\tau_0$=4ns')
   ax.legend()
   ax.grid(True, alpha=0.3)
   ax.set_aspect('equal')
   plt.show()

For dynamic FRET lines, you can use a color gradient to show the transition:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot dynamic FRET line
   fig, ax = plt.subplots(figsize=(8, 6))
   tau_f, tau_x = dynamic_fl.conversion_function
   x_values = dynamic_fl.parameter_values  # x(G,2) values
   
   # Color points by state fraction
   sc = ax.scatter(tau_f, tau_x, c=x_values, cmap='viridis', s=50, alpha=0.8)
   ax.plot(tau_f, tau_f, 'r--', linewidth=1, label='1:1 line')
   ax.set_xlabel(r'$\tau_F$ (Fluorescence-avg lifetime) [ns]')
   ax.set_ylabel(r'$\tau_X$ (Species-avg lifetime) [ns]')
   ax.set_title('Dynamic FRET Line\n$R_1$=40Å, $R_2$=80Å\n$\sigma$=6Å, $\tau_0$=4ns')
   ax.legend()
   ax.grid(True, alpha=0.3)
   ax.set_aspect('equal')
   
   # Add colorbar
   cbar = fig.colorbar(sc, ax=ax)
   cbar.set_label('x(G,2) (State 2 fraction)')
   plt.show()

Interpreting FRET Lines
----------------------

**Static FRET Line Interpretation:**

- **Distance range**: The x-axis represents the mean donor-acceptor distance :math:`R`
- **Bend direction**: The line bends **away** from the 1:1 line as FRET efficiency increases
- **At small R**: High FRET efficiency, large deviation from 1:1 line
- **At large R**: Low FRET efficiency, approaches 1:1 line
- **Slope**: The curvature depends on :math:`R_0` and :math:`\sigma`

**Dynamic FRET Line Interpretation:**

- **Fraction range**: The parameter values represent the fraction of state 2 (:math:`x_{G,2}`)
- **Endpoints**: 
  - At :math:`x_{G,2} = 0`: Pure state 1, lifetimes correspond to :math:`R_1`
  - At :math:`x_{G,2} = 1`: Pure state 2, lifetimes correspond to :math:`R_2`
- **Nonlinear transition**: The line is curved due to the nonlinear relationship between 
  lifetimes and species fractions
- **Color gradient**: Visualizes the continuous transition between states

Practical Applications
---------------------

1. **Model Validation**
   
   FRET lines can be used to validate whether a particular FRET model (static vs. dynamic) 
   is appropriate for your data. Experimental data points that fall on a static FRET line 
   suggest a single conformational state, while points falling on a dynamic FRET line 
   suggest conformational exchange.

2. **Parameter Estimation**
   
   By comparing experimental data to theoretical FRET lines, you can estimate:
   - Mean donor-acceptor distances
   - Distribution widths
   - Species fractions in multi-state systems

3. **Experimental Design**
   
   FRET lines help in designing experiments by predicting the expected lifetime 
   relationships for different distance distributions.

4. **Data Visualization**
   
   Plotting experimental data on top of FRET lines provides an intuitive visualization 
   of the conformational states present in your sample.

Example: Complete Analysis Workflow
-----------------------------------

Here's a complete example showing how to create, calculate, and visualize both types of 
FRET lines:

.. code-block:: python

   """Complete FRET Lines Example"""
   import numpy as np
   import matplotlib.pyplot as plt
   import chisurf.core.models.tcspc.fret
   import chisurf.core.fluorescence.fret.fret_line

   # Set up the R_DA axis
   chisurf.core.models.tcspc.fret.rda_axis = np.logspace(
       start=np.log(1),
       stop=np.log(500)
   )

   # Create static FRET line (R0=52, sigma=6, tau0=4)
   static_fl = chisurf.core.fluorescence.fret.fret_line.StaticFRETLine(
       n_points=100,
       parameter_range=(10, 100)
   )
   static_fl.sigma = 6.0
   static_fl.update()

   # Create dynamic FRET line (two states: 40Å and 80Å)
   dynamic_fl = chisurf.core.fluorescence.fret.fret_line.DynamicFRETLine(
       distance_1=40.0,
       distance_2=80.0,
       sigma_1=6.0,
       sigma_2=6.0,
       n_points=100,
       parameter_range=(0, 1)
   )
   dynamic_fl.update()

   # Create comparison plot
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))

   # Static FRET line plot
   ax1 = axes[0]
   tau_f_s, tau_x_s = static_fl.conversion_function
   ax1.plot(tau_f_s, tau_x_s, 'b-', linewidth=2, label='Static FRET Line')
   ax1.plot(tau_f_s, tau_f_s, 'r--', linewidth=1, label='1:1 line')
   ax1.set_xlabel(r'$\tau_F$ [ns]')
   ax1.set_ylabel(r'$\tau_X$ [ns]')
   ax1.set_title('Static FRET Line\n$R_0$=52Å, $\sigma$=6Å')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_aspect('equal')

   # Dynamic FRET line plot
   ax2 = axes[1]
   tau_f_d, tau_x_d = dynamic_fl.conversion_function
   x_vals = dynamic_fl.parameter_values
   sc = ax2.scatter(tau_f_d, tau_x_d, c=x_vals, cmap='viridis', s=50)
   ax2.plot(tau_f_d, tau_f_d, 'r--', linewidth=1, label='1:1 line')
   ax2.set_xlabel(r'$\tau_F$ [ns]')
   ax2.set_ylabel(r'$\tau_X$ [ns]')
   ax2.set_title('Dynamic FRET Line\n$R_1$=40Å, $R_2$=80Å')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   ax2.set_aspect('equal')
   fig.colorbar(sc, ax=ax2, label='x(G,2)')

   plt.tight_layout()
   plt.savefig('fret_lines_comparison.png', dpi=150)
   plt.show()

   # Print some information
   print("Static FRET Line:")
   print(f"  R0 = {static_fl.model.parameter_dict['R0'].value} Å")
   print(f"  tau0 = {static_fl.model.parameter_dict['t0'].value} ns")
   print(f"  sigma = {static_fl.sigma} Å")
   
   print("\nDynamic FRET Line:")
   print(f"  R0 = {dynamic_fl.model.parameter_dict['R0'].value} Å")
   print(f"  tau0 = {dynamic_fl.model.parameter_dict['t0'].value} ns")
   print(f"  State 1: R = {dynamic_fl.mean_distance_1} Å, sigma = {dynamic_fl.sigma_1} Å")
   print(f"  State 2: R = {dynamic_fl.mean_distance_2} Å, sigma = {dynamic_fl.sigma_2} Å")

Theoretical Background
---------------------

**FRET Efficiency and Distance:**

The FRET efficiency :math:`E` for a donor-acceptor pair separated by distance :math:`r` is:

.. math::
   E(r) = \frac{R_0^6}{R_0^6 + r^6}

For a distribution of distances :math:`P(r)`, the average efficiency is:

.. math::
   \langle E \rangle = \int_0^\infty E(r) P(r) dr

**Lifetime Relationships:**

The donor fluorescence lifetime in the presence of FRET is:

.. math::
   \tau_D = \tau_0 \cdot (1 - \langle E \rangle)

For a Gaussian distribution of distances with mean :math:`\mu` and standard deviation :math:`\sigma`:

.. math::
   P(r) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(r - \mu)^2}{2\sigma^2}\right)

The species-averaged lifetime is:

.. math::
   \tau_X = \langle \tau_D \rangle = \tau_0 \cdot \langle 1 - E \rangle

The fluorescence-averaged lifetime is:

.. math::
   \tau_F = \frac{\langle \tau_D^2 \rangle}{\langle \tau_D \rangle}

For multi-state systems, these averages are computed as weighted sums over all states.

Troubleshooting
--------------

**Common Issues:**

1. **Missing R_DA axis**: Ensure the R_DA axis is set before creating FRET lines:
   
   .. code-block:: python
   
      chisurf.core.models.tcspc.fret.rda_axis = np.logspace(
          start=np.log(1),
          stop=np.log(500)
      )

2. **AttributeError: module 'chisurf.core.fitting' has no attribute 'fit'**:
   
   This indicates the `fit` submodule is not imported. Make sure to import it:
   
   .. code-block:: python
   
      import chisurf.core.fitting.fit

3. **KeyError: 's(G,1)'**: This means the Gaussian component parameters were not properly 
   initialized. Call `find_parameters()` after appending gaussians.

4. **SVD convergence errors**: These can occur with certain parameter combinations. Try 
   adjusting the distance range or number of points.

References
----------

For a detailed theoretical treatment of FRET and FRET lines, see:

1. Peulen, T. O., Opanasyuk, O., & Seidel, C. A. M. (2017). Combining Graphical and Analytical 
   Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled 
   Macromolecules Accurately. *J. Phys. Chem. B.*, 121(35), 8211-8241.

2. Kalinin, S., Valeri, A., Antonik, M., Felekyan, S., & Seidel, C. A. (2010). Detection of 
   structural dynamics by FRET: a photon distribution and fluorescence lifetime analysis of 
   systems with multiple states. *J. Phys. Chem. B.*, 114(23), 7983-7995.

3. Hellenkamp, B., et al. (2018). Precision and accuracy of single-molecule FRET measurements—a 
   multi-laboratory benchmark study. *Nat. Methods*, 15(9), 669-676.

See Also
--------

- :ref:`Discrete FRET rate constants <discrete_fret_rate_constants>`
- :ref:`Influence of FRET efficiency <influence_of_fret_efficiency>`
- :ref:`Fluorescence Lifetime <fluorescence_lifetime>`
- :ref:`Reference curves <reference_curves>`

.. _fret_lines_example:

Example Script
--------------

A complete runnable example is available at ``examples/fret_lines_example.py`` in the 
ChiSurf repository. This script demonstrates:

- Creating both static and dynamic FRET lines
- Setting parameters (R0=52, sigma=6, tau0=4)
- Calculating the conversion functions
- Generating plots
- Interpreting the results

To run the example:

.. code-block:: bash

   cd /path/to/chisurf
   python examples/fret_lines_example.py

This will generate a plot file ``fret_lines_example.png`` showing both static and dynamic 
FRET lines.
