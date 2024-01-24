#!/usr/bin/env python3
"""
Functions and Utilities file
- Calculation Helpers
"""
# Author : Theppasith N. <tutorgaming@gmail.com
# Date : 04-June-2023

#####################################################################
# Imports
#####################################################################
import torch
import torch.nn.functional as F

#####################################################################
# Pairwise Distances
#####################################################################
def _pairwise_distance(x, squared=False, eps=1e-16):
    """
    Compute the 2D Matrix of distance between all the embeddings using
    Inner Product and Euclidean Dist Relation
     => (dist(X,Y)^2)/2 = 1 - InnerProduct(X,Y)
    """
    # Create Inner Product <X,Y> = <Embedding, Embeddings.T>
    # called Correlation Matrix
    cor_mat = torch.matmul(x, x.t())
    # Create Norm Matrix from correlation
    norm_mat = cor_mat.diag()
    # Create inner product <X,X>, <Y,Y>
    xx = norm_mat.unsqueeze(0)
    yy = norm_mat.unsqueeze(1)
    # Calculate DistanceSquared d^2 = <X,X> + <Y,Y> - 2<X,Y>
    distances = xx + yy - 2 * cor_mat
    # Filter Negative Dist
    distances = F.relu(distances)

    # We need to take square root if squared is False
    # torch.sqrt issue with Backprop - need to add epsilon for derivative
    # https://discuss.pytorch.org/t/57373
    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances

#####################################################################
# DATA Searching
#####################################################################
def _get_anchor_positive_triplet_mask(labels):
    """
    Given labels of size (batch_size, )
    Return a 2D Mask : Where mask[Anchor, Positive] is True
    iff Anchor and Positive
    - Distinct
    - Same Label

    Mask will omitted the self index (result 0)
    label [1,1,2,2,3,1]
    for item idx 0 will return [X(zero),1,0,0,0,1]
    """
    # Device Select
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Negative Mask
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    # Masked Self Index out to 0
    mask = indices_not_equal * labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (~i_equal_k)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask
