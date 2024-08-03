#!/usr/bin/env python3
"""
Jupyter Notebook Display Utilities
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(batch, classes, header=f"Visualize Batch"):
    """
    Visualize Batch of Images

    Args:
        batch (Tensor): Batch of Images
        classes (List): List of Classes
        header (str): Header
    """
    # Create a new figure
    fig = plt.figure(header)

    # Get Batch Size
    batch_size = len(batch)

    # Initialize a figure
    figsize = (batch_size, batch_size)
    print(figsize)

    # Loop through the display slot
    for i in range(0, batch_size):
        # Create Subplot for every image in this batch
        plt.subplot(2, int(batch_size/2.0), i+1)
        image = batch[0][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255.0).astype("uint8")
        # grab the label id and get the label from the classes list
        idx = batch[1][i]
        label = classes[idx]
        # show the image along with the label
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
    # show the plot
    plt.tight_layout()
    plt.show()