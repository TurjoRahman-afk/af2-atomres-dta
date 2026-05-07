import torch 
import torch.nn.functional as F 
import math 


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1]-grid_range[0]) / grid_size
        grid = (
            (
                torch.arrange(-spline_order, grid_size + spline_order + 1)
                *h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        
        self.register_buffer("grid", grid)

        