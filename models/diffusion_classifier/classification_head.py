import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.diffusion_classifier.resnet_3d import BasicBlock as Block_3d, Block_2d

UNET_BLOCK_CHANNEL = [64,64,64,64,128,128,256,256,512,512,512,512,256,256,128,128,64,64]
UNET_SPATIENT_DIM  = [1,1,2,2,4,4,8,8,8,8,8,8,4,4,2,2,1,1]

class Conv3DHead(nn.Module):
    """
    Plain (non-residual) 3D convolutional feature extractor.
 
    Mirrors the interface of Resnet3DHead but uses a simple stack of
    Conv3d -> GroupNorm -> ReLU -> Dropout3d stages instead of residual
    blocks. Useful as a lower-capacity baseline when training data /
    subject count is limited and a full ResNet3D is likely to overfit.
    """
 
    def __init__(self,
                 input_dim: int,
                 planes_3d_blocks: tuple | list,
                 pool_size: int,
                 dropout_prob: float,
                 hidden_dim: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 block_to_use: int = 13):
        super().__init__()
 
        self.in_channels = UNET_BLOCK_CHANNEL[block_to_use]
        self.hidden_dim  = hidden_dim
        layers = []
        prev_channels = UNET_BLOCK_CHANNEL[block_to_use]
        padding = kernel_size // 2

        spatial_dim = input_dim

        for stage, ch in enumerate(planes_3d_blocks):
            layers.append(
                nn.Conv3d(
                    prev_channels,
                    ch,
                    kernel_size=(kernel_size,kernel_size,kernel_size),
                    stride=(stride,stride,stride),
                    padding=(padding, padding, padding),
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm3d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout3d(p=dropout_prob))
            prev_channels = ch
            spatial_dim //= 2

        print(spatial_dim)
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.avgpool = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))
        self.fc = nn.Linear(prev_channels * (pool_size ** 3), hidden_dim)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        # print(x.shape)
        # x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)

class Resnet3DHead(nn.Module):
    def __init__(self, 
                input_dim : int,
                planes_3d_blocks : tuple | list,
                pool_size : int,
                dropout_prob : float,
                num_blocks : int,
                hidden_dim : int,
                stride: int = 2,
                block_to_use: int = 13):
        
        super().__init__()
        self.in_channels = UNET_BLOCK_CHANNEL[block_to_use]
        self.output_3d_layer = None
        self.hidden_dim = hidden_dim
        resnet_3d_layer = []

        for stage, plane in enumerate(planes_3d_blocks):
            stage_stride = stride
            resnet_3d_layer.append(
                self._make_layer(
                    ResBlock = Block_3d,
                    num_blocks = num_blocks,
                    plane = plane,
                    stride = stage_stride,
                    is_2d=False,
                    p=dropout_prob
            ))
            prev_channel = plane
            
        resnet_block_3d = nn.Sequential(*resnet_3d_layer)
        avgpool         = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))
        last            = nn.Linear(prev_channel * (pool_size ** 3), hidden_dim)

        self.output_3d_layer = nn.Sequential(
            resnet_block_3d,
            avgpool,
            nn.Flatten(start_dim=1),
            last,
        )

    def forward(self, x):
        return self.output_3d_layer(x)
    
    def _make_layer(self, ResBlock, num_blocks, plane, stride=1 , is_2d=True, p=0.5):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != plane*ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, plane * ResBlock.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm3d(plane * ResBlock.expansion)
            )
        
        layers.append(ResBlock(self.in_channels, plane, downsample = downsample, stride = stride,p=p))
        self.in_channels = plane * ResBlock.expansion

        for i in range(num_blocks - 1):
            layers.append(ResBlock(self.in_channels, plane, p=p))

        return nn.Sequential(*layers)
    








from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion

class DiffusionEncoder(nn.Module):
    def __init__(self,
                image_size, 
                diffusion: GaussianDiffusion,
                timestep : int,
                block_to_use : int,  # range from 0-17
                pool_size : int = 16,
                num_noise_samples: int = 1,
                eval_seed: int = 0,
                batch_invariant_eval: bool = True):
        '''
        Generate Frame-wise feature extracted from the diffusion model
        '''
        
        super().__init__()

        self.image_size = image_size
        self.timestep = timestep
        self.block_to_use = block_to_use
        self.num_noise_samples = num_noise_samples
        self.eval_seed = eval_seed
        self.batch_invariant_eval = batch_invariant_eval
        
        print("=> Freeze Diffusion Model Parameters!")
        self.diffusion = diffusion
        for param in self.diffusion.parameters():
            param.requires_grad = False
    

        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.LayerNorm((UNET_BLOCK_CHANNEL[block_to_use], pool_size, pool_size)),
        )

    def forward(self, x_frame):
        with torch.no_grad():
            if self.training:
                # training: average over several noise draws to reduce
                # (not eliminate) per-step variance in the gradient signal
                feat = self.diffusion.get_unet_feature_at(x_frame,
                                                           t=self.timestep,
                                                           block_idx=self.block_to_use,
                                                           num_noise_samples=self.num_noise_samples)
            else:
                # eval: fixed seed -> reproducible features for the same
                # input, so validation AUC reflects the model, not the
                # random noise draw. Relies on the val DataLoader being
                # unshuffled (shuffle=False, drop_last=False) so the same
                # clip lands in the same batch position every epoch.
                feat = self.diffusion.get_unet_feature_at(x_frame,
                                                           t=self.timestep,
                                                           block_idx=self.block_to_use,
                                                           deterministic=True,
                                                           seed=self.eval_seed,
                                                           batch_invariant=True)

        out = self.conv(feat)
        return out




