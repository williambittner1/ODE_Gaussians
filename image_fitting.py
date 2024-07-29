import math
import torch
from torch import optim
import time

from gsplat import rasterization



class SimpleTrainer:

    def __init__(self, gt_image: torch.Tensor, num_points: int = 2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H = gt_image.shape[0]
        self.W = gt_image.shape[1]
        # self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.focal = 1.0  # Set to a constant value for orthographic projection
        self.image_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        bounding_dimension = 2
        num_color_channels = 3

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.means = bounding_dimension * (torch.rand(self.num_points, 3, device = self.device) - 0.5 )
        self.scales = torch.rand(self.num_points, 3, device = self.device)
        self.rgbs = torch.rand(self.num_points, num_color_channels, device=self.device)
        self.opacities = torch.ones((self.num_points), device=self.device)
        self.background = torch.zeros(num_color_channels, device=self.device)
        self.viewmat = torch.tensor([   [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 8.0],
                                        [0.0, 0.0, 0.0, 1.0]    ], device=self.device)
        self.quats = torch.cat([    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                                    torch.sqrt(u)       * torch.sin(2.0 * math.pi * w),
                                    torch.sqrt(u)       * torch.cos(2.0 * math.pi * w)       ],  dim=-1)

        self.rgbs.requires_grad = True
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.opacities.requires_grad = True
        self.quats.requires_grad = True
        self.viewmat.requires_grad = False


    def train(self, iterations: int=1000, lr: float=0.01, save_imgs: bool=False):
        optimizer = optim.Adam([self.rgbs, self.means, self.scales, self.opacities, self.quats], lr=lr)
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2   # rasterization, backward

        K = torch.tensor([[self.focal,  0,          self.W / 2],
                            [0,         self.focal, self.H / 2],
                            [0,         0,          1         ]], device=self.device)

        for iter in range(iterations):
            start = time.time()
            renders, _, _ = rasterization(  self.means, 
                                            self.quats / self.quats.norm(dim=-1, keepdim=True),
                                            self.scales,
                                            torch.sigmoid(self.opacities),
                                            torch.sigmoid(self.rgbs),
                                            self.viewmat[None],
                                            K[None],
                                            self.W,
                                            self.H,
                                            packed=False)
