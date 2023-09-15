import torch
import torch.nn as nn

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.cosine_sim = torch.nn.CosineSimilarity(dim=0)

    def _compute_abs_similarity_sum(self, tensor, mean_tensor):
        """Compute the sum of absolute cosine similarities for each row in tensor against mean_tensor."""
        similarities = self.cosine_sim(tensor, mean_tensor.unsqueeze(0))
        return similarities.abs().sum()

    def forward(self, x):
        num_ids, num_aus, _ = x.shape

        # Calculate AU similarity
        AU_mean = x.mean(dim=1)  # Mean across IDs
        AUsim = sum(self._compute_abs_similarity_sum(x[id], AU_mean[id]) for id in range(num_ids))

        # Calculate ID similarity
        ID_mean = x.mean(dim=0)  # Mean across AUs
        IDsim = sum(self._compute_abs_similarity_sum(x[:, au], ID_mean[au]) for au in range(num_aus))

        return torch.log(AUsim / IDsim)


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.Tau = 2
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, feature_embed, label_enc, labels):
        batch_size = len(labels)
        loss_inbatch = 0

        for i in range(batch_size):
            if torch.sum(labels[i]) != 0:  # Skip if there are no positive labels

                anchor = feature_embed[i]
                sum_sim = self.cosine_similarity(anchor.unsqueeze(0), label_enc).exp().sum() / self.Tau

                # Compute negative log of positive similarities normalized by sum_sim
                pos_indices = labels[i].nonzero(as_tuple=True)[0]
                pos_sims = self.cosine_similarity(anchor.unsqueeze(0), label_enc[pos_indices]).exp() / self.Tau
                pos_loss = -torch.log(pos_sims / sum_sim).sum() / pos_sims.size(0)

                loss_inbatch += pos_loss

        return loss_inbatch / batch_size


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
        self.sim = torch.nn.CosineSimilarity()

    def forward(self,x,y):

        return abs(self.sim(x,y)).mean()