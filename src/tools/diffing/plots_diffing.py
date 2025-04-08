import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from scipy.interpolate import griddata



def plot_correlation_heatmap(correlation_data, output_dir, model_iteration, data_iteration):
    """Plot a heatmap of the top correlations."""
    correlations = correlation_data['correlations']
    top_indices = np.abs(correlations).sum(axis=1).argsort()[-100:][::-1]

    top_correlations = correlations[top_indices]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(top_correlations, cmap="coolwarm", center=0, 
                xticklabels=["pLDDT", "Activity", "TM-score"])
    plt.title("Top Feature Correlations (masked by significance)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/correlation_heatmap_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()



def plot_firing_rate_vs_correlation(correlation_data, output_dir):
    """Plot firing rates vs mean absolute correlation."""
    f_rates = correlation_data['f_rates']
    correlations = correlation_data['correlations']
    top_indices = np.abs(correlations).sum(axis=1).argsort()[-100:][::-1]
    plt.figure(figsize=(10, 10))
    plt.scatter(f_rates[top_indices], correlations[top_indices,1], cmap="coolwarm")
    plt.xlabel("Firing Rates")
    plt.ylabel("Activity")
    plt.title("Firing Rates vs Activity")
    plt.savefig(f"{output_dir}/figures/firing_rates_activity_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()


def plot_2d_density(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_dir):
    """Create a 2D density plot for CS and firing rate, colored by mean absolute correlation."""
    # Get data
    
    # Remove any NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Create a grid for the 2D density estimation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    # Create a kernel density estimate for x and y
    xy_data = np.vstack([x, y])
    kde = gaussian_kde(xy_data)
    
    # Evaluate the density on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    density = kde(positions).reshape(X.shape)
    
    # Create a 2D interpolation of z values (mean absolute correlation)
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=np.nan)
    
    # Create the interactive 2D plot
    fig = go.Figure()
    
    # Add the density contours
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=density,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorscale='Blues',
        showscale=False,
        opacity=0.5,
        name='Density'
    ))
    
    # Add the mean absolute correlation as a colored heatmap
    fig.add_trace(go.Heatmap(
        x=xi, y=yi, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Mean Absolute Correlation'),
        opacity=0.8,
        name='Mean Abs Correlation'
    ))
    
    # Update layout
    fig.update_layout(
        title='2D Density Plot: Firing Rates vs CS (colored by Mean Absolute Correlation)',
        xaxis_title='Firing Rates',
        yaxis_title='CS Values',
        width=900,
        height=700
    )
    
    # Save as interactive HTML
    fig.write_html(f"{output_dir}/figures/2d_density_plot_interactive_M{model_iteration}_D{data_iteration}.html")
    
    # Also save a static image for reference
    fig.write_image(f"{output_dir}/figures/2d_density_plot_M{model_iteration}_D{data_iteration}.png", scale=2)


def plot_3d_density(correlation_data, output_dir):
    """Create an interactive 3D density plot."""
    # Get data
    x = correlation_data['f_rates']  # Firing rates
    y = correlation_data['cs']       # CS values
    z = correlation_data['mean_abs_corr']  # Mean absolute correlation
    
    # Remove any NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Create a grid of points for the density estimation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    zi = np.linspace(z.min(), z.max(), 50)
    
    # Create 3D grid
    X, Y = np.meshgrid(xi, yi)
    
    # Create a kernel density estimate for all three dimensions
    xyz_data = np.vstack([x, y, z])
    kde = gaussian_kde(xyz_data)
    
    # For each x,y point, find the z value with maximum density
    Z = np.zeros_like(X)
    density_values = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Create a line of points along z-axis at this x,y coordinate
            points = np.vstack([np.full(len(zi), X[i,j]), 
                               np.full(len(zi), Y[i,j]), 
                               zi])
            # Evaluate density along this line
            density_along_z = kde(points)
            # Find z with maximum density
            max_idx = np.argmax(density_along_z)
            Z[i,j] = zi[max_idx]
            density_values[i,j] = density_along_z[max_idx]
    
    # Create the interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=density_values,
        colorscale='Viridis',
        colorbar=dict(title='Density')
    )])
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Density Surface: Firing Rates, CS, and Mean Absolute Correlation',
        scene=dict(
            xaxis_title='Firing Rates',
            yaxis_title='CS Values',
            zaxis_title='Mean Absolute Correlation',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    # Save as interactive HTML
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    fig.write_html(f"{output_dir}/figures/3d_density_plot_interactive_M{model_iteration}_D{data_iteration}.html")
    
    # Also save a static image for reference
    fig.write_image(f"{output_dir}/figures/3d_density_plot.png", scale=2)