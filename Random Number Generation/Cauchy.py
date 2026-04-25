# Cauchy Distribution Sampling Implementation

from __future__ import annotations  # Enable Python 3.7+ type hinting
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy.typing import ArrayLike  # Type hint for array-like inputs

def cauchy_pdf(x: ArrayLike) -> np.ndarray | float:
    """
    Calculate the Probability Density Function (PDF) of the Cauchy distribution.

    Mathematical Formula:
    -------------------
    p(x) = 1/(π(1 + x²))

    This is a special case of the general Cauchy distribution with:
    - Location parameter μ = 0 (center of symmetry)
    - Scale parameter γ = 1 (width of the distribution)

    Parameters
    ----------
    x : ArrayLike
        Input value(s) at which to evaluate the Cauchy PDF.
        Can be a single number or numpy array.

    Returns
    -------
    np.ndarray | float
        The computed PDF values. Returns float for scalar input,
        numpy array for array input.

    Notes
    -----
    The Cauchy distribution has heavy tails, meaning extreme
    values are more likely than in a normal distribution.
    It is symmetric around x = 0 and has infinite variance.
    """
    # Use np.square for better numerical stability and vectorization
    return (1.0 / np.pi) / (1.0 + np.square(x))

def rejection_sampler(n_samples: int, a: float = -6.0, b: float = 6.0) -> tuple[np.ndarray, float]:
    """
    Generate samples from a truncated Cauchy distribution using the rejection method.

    

    Algorithm Description:
    --------------------
    The rejection sampling method works as follows:
    1. Generate uniform random numbers x' from [a,b]
    2. Generate uniform random numbers r from [0,1]
    3. Accept x' if r < p(x')/M, where:
       - p(x') is the Cauchy PDF at x'
       - M is the maximum value of p(x) in [a,b] (occurs at x=0)

    Theoretical Background:
    ---------------------
    The acceptance rate is theoretically equal to:
    rate = (area under curve)/(area of bounding box)
         = [arctan(b) - arctan(a)]/[π * (b-a) * p_max]

    Parameters
    ----------
    n_samples : int
        The desired number of accepted samples.
    a : float, optional
        The lower bound of the sampling interval, by default -6.0.
    b : float, optional
        The upper bound of the sampling interval, by default 6.0.

    Returns
    -------
    tuple[np.ndarray, float]
        - samples: numpy.ndarray of shape (n_samples,) containing accepted samples
        - measured_rate: float, the true measured acceptance rate (total_accepted/total_proposals)

    Notes
    -----
    The method uses vectorized operations for efficiency, processing
    samples in batches. The batch size is chosen to balance memory
    usage with computational efficiency.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if a >= b:
        raise ValueError("Lower bound 'a' must be less than upper bound 'b'")

    p_max = cauchy_pdf(0)
    
    # Use a fixed, efficient batch size
    batch_size = 100_000 
    
    accepted_samples = []
    total_proposals = 0
    total_accepted = 0 

    print(f"Generating {n_samples:,} samples...")
    while len(accepted_samples) < n_samples:
        # 1. Generate a batch of proposals
        total_proposals += batch_size
        
        # Step 2: Vectorized proposal generation (x')
        x_proposals = np.random.uniform(a, b, batch_size)
        
        # Step 3: Vectorized acceptance check (r2)
        r_uniforms = np.random.rand(batch_size)
        
        acceptance_ratios = cauchy_pdf(x_proposals) / p_max
        accepted = x_proposals[r_uniforms < acceptance_ratios]
        
        total_accepted += len(accepted)
        
        # 2. Add accepted samples, but don't add more than n_samples
        samples_needed = n_samples - len(accepted_samples)
        accepted_samples.extend(accepted[:samples_needed])
            
    measured_rate = total_accepted / total_proposals
    
    # Return exactly n_samples
    return np.array(accepted_samples), measured_rate

def plot_results(samples: np.ndarray, measured_rate: float, a: float, b: float, theoretical_rate: float) -> None:
    """
    Visualize the sampling results against the theoretical distribution.

    This function creates a comprehensive visualization comparing the
    sampled distribution with the theoretical Cauchy distribution using
    multiple visualization techniques:

    Visualization Components:
    ----------------------
    1. Histogram of sampled data (blue bars)
       - Shows the empirical distribution of accepted samples
       - Normalized to represent probability density

    2. Theoretical PDF curve (black line)
       - The true Cauchy probability density function
       - Serves as the target distribution

    3. Kernel Density Estimation (green dashed line)
       - Non-parametric density estimation from samples
       - Smooths the histogram for better visualization
       - Validates sampling quality

    4. Proposal envelope (red dashed line)
       - Shows the uniform proposal distribution's maximum
       - Illustrates the rejection sampling envelope

    Statistical Metrics:
    ------------------
    - Measured acceptance rate: Actual proportion of accepted samples
    - Theoretical rate: Expected proportion based on area ratio
    - Efficiency: Ratio of measured to theoretical rates

    Parameters
    ----------
    samples : numpy.ndarray
        Array of samples generated from the rejection sampler.
    measured_rate : float
        The measured efficiency of the sampler.
    a : float
        Lower bound of the sampling interval.
    b : float
        Upper bound of the sampling interval.
    theoretical_rate : float
        The theoretically calculated acceptance rate.
    """
    p_max = cauchy_pdf(0)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    print("Calculating Kernel Density Estimation (KDE)...")
    kde = gaussian_kde(samples)
    x_theory = np.linspace(a, b, 500)
    y_kde = kde(x_theory)
    
    # --- Plotting ---
    
    # 1. Plot the histogram
    # We use np.histogram to get the bin heights *before* plotting
    # This is crucial for setting the y-limit
    hist_heights, bin_edges = np.histogram(samples, bins=100, density=True)
    
    plt.hist(samples, bins=100, density=True, color='#4A90E2', alpha=0.7,
             label=f'Histogram of {len(samples):,} Samples')

    # 2. Plot the theoretical PDF
    y_theory = cauchy_pdf(x_theory)
    plt.plot(x_theory, y_theory, 'k-', linewidth=3,
             label='Theoretical Cauchy PDF (Target)')

    # 3. Plot the KDE
    plt.plot(x_theory, y_kde, 'g--', linewidth=2.5,
             label='KDE from Samples (Data-Driven)')

    # 4. Plot the proposal envelope
    plt.plot([a, b], [p_max, p_max], 'r--', linewidth=2,
             label=f'Proposal Envelope ($p_{{max}}$)')
    
    # Find the maximum value from all plotted elements
    max_hist = np.max(hist_heights)
    max_kde = np.max(y_kde)
    
    # Our upper limit is the largest of these, plus 10% padding
    plot_top = max(p_max, max_hist, max_kde) * 1.10
    
    # Set the y-axis limits dynamically
    plt.xlim(a, b)
    plt.ylim(0, plot_top)

    # Add text annotation
    info_text = (
        f"Measured Rate: {measured_rate:.4%}\n"
        f"Theoretical Rate: {theoretical_rate:.4%}\n"
        f"Efficiency: {measured_rate / theoretical_rate:.2%}"
    )
    # Place text relative to the new dynamic top
    plt.text(a * 0.98, plot_top * 0.95, info_text,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc='wheat', alpha=0.7))

    plt.title('Rejection Sampling of the Cauchy Distribution', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density p(x)', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.show()

def plot_theoretical_pdf(a: float, b: float) -> None:
    """
    Plots the theoretical Cauchy PDF in isolation.
    Colors are chosen based on the latest request (pinkish inside curve, bluish outside curve).

    Parameters
    ----------
    a : float
        Lower bound of the plotting interval.
    b : float
        Upper bound of the plotting interval.
    """
    print("Generating theoretical PDF plot...")
    
    plt.style.use('default') 
    fig = plt.figure(figsize=(8, 5), facecolor='#e0f2f7') 
    ax = fig.add_subplot(111)
    ax.set_facecolor('#cceeff') 
    x_domain = np.linspace(a, b, 400)
    y_pdf = cauchy_pdf(x_domain)
    
    # Fill the area UNDER the theoretical PDF curve with a pinkish color
    plt.fill_between(x_domain, y_pdf, color='#ffccdd', alpha=0.8, label='Area under PDF (Pinkish)') # Pinkish inside the curve
    
    # Plot the theoretical PDF curve itself (e.g., in a dark blue for definition)
    plt.plot(x_domain, y_pdf, color='#000080', linewidth=2.5, 
             label='$p(x) = \\frac{1}{\\pi(1 + x^2)}$')
    
    plt.title('Theoretical Cauchy Distribution PDF', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density p(x)', fontsize=12)
    
    # Get the peak for setting a clean y-limit
    p_max = cauchy_pdf(0)
    plt.xlim(a, b)
    plt.ylim(0, p_max * 1.1) # Ensure space for the peak
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, color='gray') # Keep grid and set its color
    
    plt.show()
    
# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating the Cauchy distribution sampling.
    
    This section:
    1. Sets up the sampling parameters
    2. Calculates theoretical acceptance rate
    3. Performs the sampling
    4. Visualizes and compares results
    
    The bounds (-6, 6) are chosen to capture most of the distribution's
    mass while keeping the acceptance rate reasonable. The Cauchy
    distribution has heavy tails, so we can't capture all of it.
    """
    # Configuration
    NUM_SAMPLES = 100_000  # Number of samples to generate
    LOWER_BOUND = -6.0     # Chosen to capture ~86% of distribution
    UPPER_BOUND = 6.0      # Symmetric bounds around 0
    
    # Theoretical Rate = Area under target curve / Area of proposal box
    p_max_val = cauchy_pdf(0)
    area_proposal_box = (UPPER_BOUND - LOWER_BOUND) * p_max_val
    
    # Integral of (1/pi) * 1/(1+x^2) is (1/pi) * arctan(x)
    area_curve = (1.0 / np.pi) * (np.arctan(UPPER_BOUND) - np.arctan(LOWER_BOUND))
    
    theoretical_rate = area_curve / area_proposal_box
    
    try:
        final_samples, measured_efficiency = rejection_sampler(
            n_samples=NUM_SAMPLES,
            a=LOWER_BOUND,
            b=UPPER_BOUND
        )
        
        print(f"\nSampling complete.")
        print(f"  Theoretical acceptance rate: {theoretical_rate:.4%}")
        print(f"  Measured acceptance rate:    {measured_efficiency:.4%}")

        choice = int(input("\nChoose plot to display:\n1. Theoretical PDF\n2. Rejection Plot\nChoice (1/2): "))
        if choice == 1:
            # 1. Show the isolated theoretical plot
            plot_theoretical_pdf(a=LOWER_BOUND, b=UPPER_BOUND)
        elif choice == 2:
            # 2. Show the main analysis plot
            plot_results(final_samples, measured_efficiency,
                        a=LOWER_BOUND, b=UPPER_BOUND,
                        theoretical_rate=theoretical_rate)
        else :
            print("Invalid choice. Please enter 'theoretical' or 'analysis'.")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")