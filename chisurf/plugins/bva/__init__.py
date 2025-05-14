"""
Single Molecule Burst Variance Analysis (BVA) Plugin

This plugin implements Burst Variance Analysis for single-molecule FRET experiments.
BVA is a technique that analyzes the variance of FRET efficiency within individual
bursts to distinguish between static and dynamic heterogeneity in the sample.

The plugin allows users to:
1. Load burst data from previous burst analysis
2. Configure BVA parameters (donor/acceptor channels, micro-time ranges, etc.)
3. Compute BVA metrics for each burst
4. Visualize results with static FRET line for comparison
5. Save results to files for further analysis

BVA is particularly useful for identifying conformational dynamics in biomolecules
that occur on timescales comparable to the burst duration.
"""

name = "Single-Molecule:Burst-Variance Analysis"

