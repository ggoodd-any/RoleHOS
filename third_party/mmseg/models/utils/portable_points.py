import torch.nn.functional as F

def point_sample(input, points, align_corners=False, **kwargs):
    add_dim = points.ndim == 3
    if add_dim:
        points = points.unsqueeze(2)
    result = F.grid_sample(input, 2 * points - 1, align_corners=align_corners, **kwargs)
    return result.squeeze(3) if add_dim else result
