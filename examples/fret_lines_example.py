"""Example script demonstrating Static and Dynamic FRET Lines.

This script shows how to create and visualize FRET lines using the chisurf
FRET line generators. It demonstrates:

1. Static FRET line with R0=52 Angstrom, sigma=6 Angstrom, tau0=4 ns
2. Dynamic FRET line with two states exchanging between each other

The FRET line represents the relationship between fluorescence-averaged lifetime
(tau_F) and species-averaged lifetime (tau_X) for different donor-acceptor
separation distances or species fractions.
"""

import numpy as np
import chisurf.core.models.tcspc.fret
import chisurf.core.fluorescence.fret.fret_line


def setup_rda_axis():
    """Set up the global R_DA axis for FRET calculations."""
    chisurf.core.models.tcspc.fret.rda_axis = np.logspace(
        start=np.log(1),
        stop=np.log(500)
    )


def create_static_fret_line():
    """Create and return a static FRET line with specified parameters.
    
    Parameters used:
        R0 = 52 Angstrom (Forster radius)
        sigma = 6 Angstrom (width of distance distribution)
        tau0 = 4 ns (donor lifetime without FRET)
    
    Returns:
        StaticFRETLine: The FRET line object
    """
    print("\n" + "="*70)
    print("Static FRET Line Example")
    print("="*70)
    print("Parameters: R0=52 Angstrom, sigma=6 Angstrom, tau0=4 ns")
    
    # Create StaticFRETLine with default parameters (R0=52, tau0=4 from settings)
    static_fl = chisurf.core.fluorescence.fret.fret_line.StaticFRETLine(
        n_points=100,
        parameter_range=(10, 100),  # Distance range in Angstrom
        verbose=True
    )
    
    # Set sigma to 6 Angstrom
    static_fl.sigma = 6.0
    
    # Verify the parameters
    print(f"\nModel parameters:")
    print(f"  R0 (Forster radius): {static_fl.model.parameter_dict['R0'].value} Angstrom")
    print(f"  t0 (donor lifetime): {static_fl.model.parameter_dict['t0'].value} ns")
    print(f"  sigma (distribution width): {static_fl.sigma} Angstrom")
    print(f"  k2 (orientation factor): {static_fl.model.parameter_dict['k2'].value}")
    
    # Calculate the FRET line
    print("\nCalculating static FRET line...")
    static_fl.update()
    
    # Get the conversion function data
    tau_f, tau_x = static_fl.conversion_function
    
    print(f"\nComputed {len(tau_f)} points along the FRET line")
    print(f"  tau_f (fluorescence-averaged lifetime) range: {tau_f.min():.3f} to {tau_f.max():.3f} ns")
    print(f"  tau_x (species-averaged lifetime) range: {tau_x.min():.3f} to {tau_x.max():.3f} ns")
    
    # Get the polynomial approximation of the conversion function
    coeffs = static_fl.polynom_coefficients
    print(f"\nPolynomial coefficients (degree {len(coeffs)-1}):")
    print(f"  {coeffs}")
    
    # Get string representations for plotting
    print(f"\nConversion function string:")
    print(f"  {static_fl.conversion_function_string}")
    print(f"\nTransfer efficiency string:")
    print(f"  {static_fl.transfer_efficency_string}")
    
    # Compute transfer efficiency at a few points
    print(f"\nTransfer efficiency at different distances:")
    for r in [30, 40, 52, 60, 80]:
        # Find the closest parameter value
        idx = np.argmin(np.abs(static_fl.parameter_values - r))
        eff = static_fl.fret_efficiencies[idx]
        print(f"  R = {r:5.1f} Angstrom: E = {eff:.4f}")
    
    return static_fl


def create_dynamic_fret_line():
    """Create and return a dynamic FRET line with two exchanging states.
    
    This demonstrates a system with two conformational states:
        State 1: R1 = 40 Angstrom, sigma1 = 6 Angstrom
        State 2: R2 = 80 Angstrom, sigma2 = 6 Angstrom
    
    The dynamic FRET line shows how the observed lifetimes vary as the
    population shifts between the two states.
    
    Returns:
        DynamicFRETLine: The FRET line object
    """
    print("\n" + "="*70)
    print("Dynamic FRET Line Example")
    print("="*70)
    print("Two-state system with exchange between states")
    print("  State 1: R=40 Angstrom, sigma=6 Angstrom")
    print("  State 2: R=80 Angstrom, sigma=6 Angstrom")
    print("Parameters: R0=52 Angstrom, tau0=4 ns")
    
    # Create DynamicFRETLine with two states
    dynamic_fl = chisurf.core.fluorescence.fret.fret_line.DynamicFRETLine(
        distance_1=40.0,
        distance_2=80.0,
        sigma_1=6.0,
        sigma_2=6.0,
        n_points=100,
        parameter_range=(0, 1),  # Vary x(G,2) from 0 to 1
        verbose=True
    )
    
    # Verify the parameters
    print(f"\nModel parameters:")
    print(f"  R0 (Forster radius): {dynamic_fl.model.parameter_dict['R0'].value} Angstrom")
    print(f"  t0 (donor lifetime): {dynamic_fl.model.parameter_dict['t0'].value} ns")
    print(f"  R(G,1) (state 1 mean distance): {dynamic_fl.mean_distance_1} Angstrom")
    print(f"  R(G,2) (state 2 mean distance): {dynamic_fl.mean_distance_2} Angstrom")
    print(f"  s(G,1) (state 1 sigma): {dynamic_fl.sigma_1} Angstrom")
    print(f"  s(G,2) (state 2 sigma): {dynamic_fl.sigma_2} Angstrom")
    
    # Calculate the FRET line
    print("\nCalculating dynamic FRET line...")
    dynamic_fl.update()
    
    # Get the conversion function data
    tau_f, tau_x = dynamic_fl.conversion_function
    x_values = dynamic_fl.parameter_values  # This is x(G,2), the fraction of state 2
    
    print(f"\nComputed {len(tau_f)} points along the FRET line")
    print(f"  x(G,2) (state 2 fraction) range: {x_values.min():.3f} to {x_values.max():.3f}")
    print(f"  tau_f (fluorescence-averaged lifetime) range: {tau_f.min():.3f} to {tau_f.max():.3f} ns")
    print(f"  tau_x (species-averaged lifetime) range: {tau_x.min():.3f} to {tau_x.max():.3f} ns")
    
    # Get the polynomial approximation
    coeffs = dynamic_fl.polynom_coefficients
    print(f"\nPolynomial coefficients (degree {len(coeffs)-1}):")
    print(f"  {coeffs}")
    
    # Compute transfer efficiency at different state fractions
    print(f"\nTransfer efficiency at different state 2 fractions (x(G,2)):")
    for x2 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Find the closest parameter value
        idx = np.argmin(np.abs(x_values - x2))
        eff = dynamic_fl.fret_efficiencies[idx]
        x1 = 1.0 - x2
        print(f"  x(G,2) = {x2:.2f} (x(G,1) = {x1:.2f}): E = {eff:.4f}")
    
    return dynamic_fl


def create_plot(static_fl, dynamic_fl, save_fig=False, filename=None):
    """Create a matplotlib plot comparing static and dynamic FRET lines.
    
    Parameters:
        static_fl: StaticFRETLine object
        dynamic_fl: DynamicFRETLine object
        save_fig: bool, whether to save the figure to a file
        filename: str, filename to save the figure to
    
    Returns:
        fig: matplotlib.figure.Figure object
        axes: array of matplotlib.axes.Axes objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available. Cannot create plots.")
        print("To see the FRET line plots, please install matplotlib:")
        print("  pip install matplotlib")
        return None, None
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Static FRET Line
    ax1 = axes[0]
    tau_f_static, tau_x_static = static_fl.conversion_function
    ax1.plot(tau_f_static, tau_x_static, 'b-', linewidth=2, label='Static FRET Line')
    ax1.plot(tau_f_static, tau_f_static, 'r--', linewidth=1, label='1:1 line')
    ax1.set_xlabel(r'$\tau_F$ (Fluorescence-avg lifetime) [ns]', fontsize=12)
    ax1.set_ylabel(r'$\tau_X$ (Species-avg lifetime) [ns]', fontsize=12)
    ax1.set_title('Static FRET Line\nRo=52A, sigma=6A, tau0=4ns', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add some text annotation
    ax1.text(0.05, 0.95, 
             f'R range: {static_fl.parameter_range[0]:.0f}-{static_fl.parameter_range[1]:.0f} A\n'
             f'Points: {len(static_fl.parameter_values)}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Dynamic FRET Line
    ax2 = axes[1]
    tau_f_dynamic, tau_x_dynamic = dynamic_fl.conversion_function
    x_values = dynamic_fl.parameter_values
    
    # Create a color gradient based on x(G,2) fraction
    colors = plt.cm.viridis(x_values)
    sc = ax2.scatter(tau_f_dynamic, tau_x_dynamic, c=x_values, cmap='viridis', 
                    s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax2.plot(tau_f_dynamic, tau_f_dynamic, 'r--', linewidth=1, label='1:1 line')
    ax2.set_xlabel(r'$\tau_F$ (Fluorescence-avg lifetime) [ns]', fontsize=12)
    ax2.set_ylabel(r'$\tau_X$ (Species-avg lifetime) [ns]', fontsize=12)
    ax2.set_title('Dynamic FRET Line\nR1=40A, R2=80A\nsigma=6A, tau0=4ns', 
                 fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.8)
    cbar.set_label('x(G,2) (State 2 fraction)', fontsize=12)
    
    # Add text annotation
    ax2.text(0.05, 0.95,
             f'State 1: R={dynamic_fl.mean_distance_1:.0f} A\n'
             f'State 2: R={dynamic_fl.mean_distance_2:.0f} A\n'
             f'Points: {len(dynamic_fl.parameter_values)}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_fig:
        if filename is None:
            filename = 'fret_lines_comparison.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {filename}")
    
    return fig, axes


def main():
    """Main function to run the FRET lines example."""
    print("="*70)
    print("FRET Lines Example - Static and Dynamic")
    print("="*70)
    
    # Set up the R_DA axis
    setup_rda_axis()
    
    # Create static FRET line
    static_fl = create_static_fret_line()
    
    # Create dynamic FRET line
    dynamic_fl = create_dynamic_fret_line()
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nStatic FRET Line:")
    print(f"  - Represents a single conformational state with Gaussian distance distribution")
    print(f"  - Distance range: {static_fl.parameter_range}")
    print(f"  - Sigma: {static_fl.sigma} Angstrom")
    print(f"  - Points: {len(static_fl.parameter_values)}")
    
    print("\nDynamic FRET Line:")
    print(f"  - Represents exchange between two conformational states")
    print(f"  - State 1: R={dynamic_fl.mean_distance_1} Angstrom, sigma={dynamic_fl.sigma_1} Angstrom")
    print(f"  - State 2: R={dynamic_fl.mean_distance_2} Angstrom, sigma={dynamic_fl.sigma_2} Angstrom")
    print(f"  - Fraction range: {dynamic_fl.parameter_range}")
    print(f"  - Points: {len(dynamic_fl.parameter_values)}")
    
    # Try to create the plot
    fig, axes = create_plot(static_fl, dynamic_fl, save_fig=True, 
                           filename='fret_lines_example.png')
    
    if fig is not None:
        print("\nTo view the plot, open: fret_lines_example.png")
        print("\nAlternatively, to display interactively (if running in an environment")
        print("with a display), add plt.show() to the code.")
    else:
        print("\nNote: Matplotlib is not available. The numerical results are still valid.")
    
    # Additional information
    print("\n" + "="*70)
    print("Understanding FRET Lines")
    print("="*70)
    print("""
A FRET line represents the relationship between:
  - tau_F (fluorescence-averaged lifetime): <tau^2> / <tau>
  - tau_X (species-averaged lifetime): <tau>

For an ideal single exponential donor (no FRET), tau_F = tau_X.
With FRET, the relationship deviates from the 1:1 line (red dashed line).

Static FRET Line:
  - Shows how tau_F and tau_X vary with donor-acceptor distance R
  - Assumes a Gaussian distribution of distances with width sigma
  - The line bends away from 1:1 as FRET efficiency increases

Dynamic FRET Line:
  - Shows how tau_F and tau_X vary with species fraction x(G,2)
  - Represents exchange between two conformational states
  - The line shows a nonlinear transition between the two limiting states
  - At x(G,2)=0: Pure state 1 (R1=40 Angstrom)
  - At x(G,2)=1: Pure state 2 (R2=80 Angstrom)

Key Parameters:
  - R0 (Forster radius): Distance at which FRET efficiency is 50 percent
  - sigma: Width of the Gaussian distance distribution
  - tau0: Donor lifetime in the absence of FRET
  - k2 (kappa^2): Orientation factor (typically 2/3 for isotropic averaging)
""")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
