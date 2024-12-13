import numpy as np
import torch

def hybrid_z_sampling(z, encoded_real, batch_size, z_size, device, max_attempts=10, min_distance=20.0):
    """
    Hybrid approach to generate a latent vector z that is sufficiently distant from
    the encoded real data z_encoded. First attempts to sample new z values, and if
    not successful, scales the last z.
        Arguments:
        ---------
            - z (torch.Tensor): Random noise data
            - encoded_real (torch.Tensor): Embedding from real data E(x) (output from Encoder)
            - batch_size (int): Batch size
            - z_size (int): Dimension of the latent vector z
            - device (str): The device tensors are on (CPU or GPU)
            - max_attempts (int): Maximum number of attempts to sample a new z.
            - min_distance (float): Minimum Euclidean distance required from z to z_encoded.
        Return:
        ---------
            - z (torch.Tensor): The suitable latent vector z for the generator.
    """

    for attempt in range(max_attempts):
        # Calculate distances for the batch
        differences = z - encoded_real
        distances = torch.norm(differences.view(batch_size, -1), dim=1, p=2)  # Flatten before calculating norms
        compliant_mask = distances >= min_distance
        
        if compliant_mask.all():  # If all are compliant, return
            return z
        
        # Resample where not compliant
        non_compliant_indices = ~compliant_mask
        z[non_compliant_indices] = torch.randn(non_compliant_indices.sum().item(), z_size, 1, device=device)

    # For any remaining non-compliant, apply scaling
    for i in range(batch_size):
        if not compliant_mask[i]:
            difference = z[i] - encoded_real[i]
            norm_difference = torch.norm(difference, p=2)
            required_scale = torch.clamp(min_distance / norm_difference, min=1.0)
            z[i] = encoded_real[i] + difference * required_scale

    return z

def signal_change_z_sampling(z, min_changes=20, max_changes=80):
    """
    Signal change approach to generate a latent vector z that is sufficiently distant from
    the encoded real data z_encoded. Flip the signal (multiply by -1) of some of the values in E(x) to make
    G(z) distant from x.
        Arguments:
        ---------
            - z (torch.Tensor): Embbeding from real data
        Return:
        ---------
            - z_random_signs (torch.Tensor): The suitable latent vector z for the generator.
    """
    
    changes = torch.randint(min_changes, max_changes, (1,))[0]

    indices = torch.argsort(torch.rand((z.shape[0], z.shape[1])), dim=-1)[:, :changes]

    # Create a mask for all indices to change
    mask = torch.zeros(z.shape[0], z.shape[1], dtype=torch.bool)
    mask.scatter_(1, indices, True)

    random_signs = torch.where(mask, torch.tensor(-1.0), torch.tensor(1.0)).unsqueeze(-1)

    # Multiply the original z with the random sign tensor to maintain or change signals
    z_random_signs = z * random_signs

    return z_random_signs
