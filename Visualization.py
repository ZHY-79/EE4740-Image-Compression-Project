import matplotlib.pyplot as plt

def display_recovered_image(recovered_x, original_x=None, title=None, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    
    if original_x is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(original_x.reshape(28, 28), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(recovered_x.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')
    else:
        plt.imshow(recovered_x.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed Image' if title is None else title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_recovered_image_color(recovered_x, original_x=None, title=None, figsize=(10, 5), cmap='gray'):
    
    recovered_display = recovered_x.copy()
    if recovered_display.max() - recovered_display.min() > 0:
        recovered_display = (recovered_display - recovered_display.min()) / (recovered_display.max() - recovered_display.min()) * 255
    
    plt.figure(figsize=figsize)
    
    if original_x is not None:
        original_display = original_x.copy()
        if original_display.max() - original_display.min() > 0:
            original_display = (original_display - original_display.min()) / (original_display.max() - original_display.min()) * 255
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_display.reshape(28, 28), cmap=cmap)
        plt.title(f'Original Image')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(recovered_display.reshape(28, 28), cmap=cmap)
        plt.title(f'Recovered (Auto-scaled)')
        plt.axis('off')
        plt.colorbar()
    else:
        plt.imshow(recovered_display.reshape(28, 28), cmap=cmap)
        plt.title(f'Recovered Image\nRange: [{recovered_x.min():.4f}, {recovered_x.max():.4f}]')
        plt.axis('off')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()