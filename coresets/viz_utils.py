import matplotlib.pyplot as plt

def plot_image_grid(images, labels=None, grid_size=(3, 3), figsize=(3, 3)):
    """
    Plot specific images in a grid.
    
    Parameters:
    images: array of torch images
    labels: optional array of corresponding labels
    grid_size: tuple of (rows, cols)
    figsize: figure size in inches
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes if grid_size is (1, n) or (n, 1)
    if rows == 1 or cols == 1:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # Display the image
            img = images[i]
            ax.imshow(img, cmap='gray')
            
            # Add label if provided
            if labels is not None:
                ax.set_title(f'Label: {labels[i]}')
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide empty subplots
            ax.axis('off')
    
    plt.tight_layout()
    # return fig