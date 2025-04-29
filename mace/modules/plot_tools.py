def plot_edge_data(edge_length, edge_density, edge_density_attn, save_path=None):
    """
    Plot edge length vs density/attention with log scale on x-axis.
    
    Args:
        edge_length: Tensor of shape [N, 1] containing edge lengths
        edge_density: Tensor of shape [N, 3] containing edge density values
        edge_density_attn: Tensor of shape [N, 3] containing edge attention values
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert tensors to numpy for plotting
    lengths = edge_length.cpu().detach().numpy().flatten()
    densities = edge_density.cpu().detach().numpy()
    attentions = edge_density_attn.cpu().detach().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot edge density (all three channels)
    for i in range(3):
        ax1.scatter(lengths, densities[:, i], alpha=0.5, s=2, label=f'Channel {i}')
    
    ax1.set_xscale('log')  # Set x-axis to log scale
    ax1.set_xlabel('Edge Length (log scale)')
    ax1.set_ylabel('Edge Density')
    ax1.set_title('Edge Length vs Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot edge attention (all three channels)
    for i in range(3):
        ax2.scatter(lengths, attentions[:, i], alpha=0.5, s=2, label=f'Channel {i}')
    
    ax2.set_xscale('log')  # Set x-axis to log scale
    ax2.set_xlabel('Edge Length (log scale)')
    ax2.set_ylabel('Edge Attention')
    ax2.set_title('Edge Length vs Attention')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print some statistics
    print(f"Edge length range: {lengths.min():.4f} to {lengths.max():.4f}")
    print(f"Edge density range: {densities.min():.4f} to {densities.max():.4f}")
    print(f"Edge attention range: {attentions.min():.4f} to {attentions.max():.4f}")


def plot_edge_data_message(edge_length, edge_density, edge_density_attn, mji, save_path=None):
    """
    Plot edge length vs various edge properties with log scale on x-axis.
    
    Args:
        edge_length: Tensor of shape [N, 1] containing edge lengths
        edge_density: Tensor of shape [N, 3] containing edge density values
        edge_density_attn: Tensor of shape [N, 3] containing edge attention values
        mji: Tensor of shape [N, feature_dim] containing mji features
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert tensors to numpy for plotting
    lengths = edge_length.cpu().detach().numpy().flatten()
    densities = edge_density.cpu().detach().numpy()
    attentions = edge_density_attn.cpu().detach().numpy()
    mji_np = mji.cpu().detach().numpy()
    
    # Calculate norms
    density_mag = np.linalg.norm(densities, axis=1)
    attn_mag = np.linalg.norm(attentions, axis=1)
    mji_mag = np.linalg.norm(mji_np, axis=1)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Edge Length vs Density Magnitude
    axs[0, 0].scatter(lengths, density_mag, alpha=0.3, s=1, color='blue')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('Edge Length (log scale)')
    axs[0, 0].set_ylabel('Edge Density Magnitude')
    axs[0, 0].set_title('Edge Length vs Density Magnitude')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Edge Length vs Attention Magnitude
    axs[0, 1].scatter(lengths, attn_mag, alpha=0.3, s=1, color='red')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Edge Length (log scale)')
    axs[0, 1].set_ylabel('Edge Attention Magnitude')
    axs[0, 1].set_title('Edge Length vs Attention Magnitude')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Edge Length vs mji Magnitude
    axs[1, 0].scatter(lengths, mji_mag, alpha=0.3, s=1, color='green')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel('Edge Length (log scale)')
    axs[1, 0].set_ylabel('mji Magnitude')
    axs[1, 0].set_title('Edge Length vs mji Magnitude')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Histogram of edge lengths (for context)
    axs[1, 1].hist(lengths, bins=100, alpha=0.7, color='purple')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Edge Length (log scale)')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title('Distribution of Edge Lengths')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print some statistics
    print(f"Edge length range: {lengths.min():.4f} to {lengths.max():.4f}")
    print(f"Edge density magnitude range: {density_mag.min():.4f} to {density_mag.max():.4f}")
    print(f"Edge attention magnitude range: {attn_mag.min():.4f} to {attn_mag.max():.4f}")
    print(f"mji magnitude range: {mji_mag.min():.4f} to {mji_mag.max():.4f}")
    print(f"mji feature dimension: {mji_np.shape[1]}")

