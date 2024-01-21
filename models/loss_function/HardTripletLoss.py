
import torch
import torch.nn as nn
from util.util import _pairwise_distance, _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask, _get_triplet_mask
import torch.nn.functional as F

# Hard Triplet Loss
class HardTripletLoss(nn.Module):
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        # Parameter Mode Config
        self.margin  = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # print("Embeddings Size = {}".format(embeddings.shape))
        # print("labels Size = {}".format(labels.shape))

        # Populate All Data - Pair Distances
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        # Switch Mode between hardest triplet loss or hard triplet loss
        if self.hardest:
            triplet_loss = self.hardest_triplet_loss(pairwise_dist, embeddings, labels)
            # print("Hardest Loss Shape : {}".format(triplet_loss.shape))

        else:
            triplet_loss = self.hard_triplet_loss(pairwise_dist, embeddings, labels)
            # print("Hard Loss Shape : {}".format(triplet_loss.shape))

        # print("Loss Shape : {}".format(triplet_loss.shape))
        return triplet_loss

    def hardest_triplet_loss(self, pairwise_dist, embeddings, labels):
        """
        Hardest Triplet Loss
        - Get The Hardest Positive Pair (Biggest Distance = Very Loss)
        - Get The Hardest Negative Pair (Nearest Distance = Very Loss)
        - Calculate Distance between Those Hardest and our anchor
        - Loss = Relu(hardest_pos_dist - hardest_negative + margin?)
        """
        # Find the Hardest Positive Pair
        # - Get Same Class Mask
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
        valid_positive_dist = pairwise_dist * mask_anchor_positive
        # - Find the Hardest Positive distances (Farthest = Very Loss)
        hardest_positive_dist, _ = torch.max(
            valid_positive_dist,
            dim=1, keepdim=True
        )
        # print("Shape of Hardest Pos : {}".format(hardest_positive_dist.shape))

        # Find the Hardest Negative Pair
        # - Get Negative Class Mask
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = torch.min(
            anchor_negative_dist,
            dim=1, keepdim=True
        )
        # print("Shape of Hardest Neg : {}".format(hardest_negative_dist.shape))


        # Find the Loss
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        # print("Shape of Triplet Loss : {}".format(triplet_loss.shape))

        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

    def hard_triplet_loss(self, pairwise_dist, embeddings, labels):
        anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
        anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
        # Uses broadcasting where the
        # 1st argument has shape (batch_size, batch_size, 1)
        # 2nd argument has shape (batch_size, 1, batch_size)
        loss = anc_pos_dist - anc_neg_dist + self.margin

        mask = _get_triplet_mask(labels).float()
        triplet_loss = loss * mask

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss
