import torch
import torch.nn as nn


from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from models.diffusion_classifier.classification_head import DiffusionEncoder
from models.diffusion_classifier.classification_head import Resnet3DHead, Conv3DHead

class FrozenVDEWithDemographics(nn.Module):
    """Fixed image logit plus a learned demographic/clinical correction."""
    def __init__(self, image_model, sex_dim=4, age_dim=4, ppt_dim=4):
        super().__init__()
        self.image_model = image_model.requires_grad_(False).eval()
        self.sex_embedding = nn.Embedding(2, sex_dim)
        self.age_embedding = nn.Linear(1, age_dim, bias=False)
        self.ppt_pre_embedding = nn.Linear(1, ppt_dim, bias=False)
        self.ppt_post_embedding = nn.Linear(1, ppt_dim, bias=False)
        self.demo_classifier = nn.Linear(sex_dim + age_dim + 2 * ppt_dim, 1)
        nn.init.zeros_(self.demo_classifier.weight)
        nn.init.zeros_(self.demo_classifier.bias)

    def train(self, mode=True):
        super().train(mode)
        # Also freeze BatchNorm statistics, dropout, and diffusion noise mode.
        self.image_model.eval()
        return self

    def forward(self, data):
        device = self.demo_classifier.weight.device
        with torch.no_grad():
            image_logit = self.image_model(data)
        demo = data['demo']
        ppt = demo['ppt'].to(device).float()
        features = torch.cat([
            self.sex_embedding(demo['sex'].to(device).long().reshape(-1)),
            self.age_embedding(demo['age'].to(device).float().reshape(-1, 1)),
            self.ppt_pre_embedding(ppt[:, 0:1]),
            self.ppt_post_embedding(ppt[:, 1:2]),
        ], dim=1)
        return image_logit + self.demo_classifier(features)


class VDE(nn.Module):
    def __init__(self, 
                 diffusion_ckpt: str, 
                 unet_config   : dict,
                 diffusion_config: dict,
                 encoder_config: dict, 
                 resnet_config:  dict,
                 use_demo: bool = True, 
                 sex_embedding_dim: int = 8,
                 age_embedding_dim: int = 8,
                 ppt_embedding_dim: int = 8,
                 num_ppt: int = 2, ppt_only: bool = False,
                 intervention_matched_ppt: bool = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        diffusion_model      = self.__load_diffusion_model(unet_config, diffusion_config, diffusion_ckpt)
        self.encoder         = DiffusionEncoder(diffusion=diffusion_model, **encoder_config) 
        self.head            = Conv3DHead(**resnet_config)

        self.use_demo = use_demo
        self.ppt_only = ppt_only
        self.intervention_matched_ppt = intervention_matched_ppt
        self.sex_embedding_dim = sex_embedding_dim
        self.age_embedding_dim = age_embedding_dim
        self.ppt_embedding_dim = ppt_embedding_dim

        
        last_dim = self.head.hidden_dim

        if self.use_demo:
            if not self.ppt_only:
                # sex embedding
                self.sex_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=self.sex_embedding_dim)
                last_dim += self.sex_embedding_dim
                # age embedding
                self.age_embedding_layer = nn.Sequential(nn.Linear(1, self.age_embedding_dim, bias=False))
                last_dim += self.age_embedding_dim
            # Dataset order: standardized pre-intervention, post-intervention PPT.
            if num_ppt != 2:
                raise ValueError("Separate pre/post PPT embeddings require num_ppt=2")
            self.ppt_pre_embedding_layer = nn.Linear(1, self.ppt_embedding_dim, bias=False)
            self.ppt_post_embedding_layer = nn.Linear(1, self.ppt_embedding_dim, bias=False)
            last_dim += 2 * self.ppt_embedding_dim

        self.last = nn.Linear(last_dim, 1)
    
    def __load_diffusion_model(self, unet_config, diffusion_config, diffusion_ckpt: str):
        backbone  = Unet(**unet_config)
        diffusion = GaussianDiffusion(model=backbone, **diffusion_config)
        ckpt      = torch.load(diffusion_ckpt, map_location=self.device, weights_only=True)
        diffusion.load_state_dict(ckpt['model'])

        for param in diffusion.parameters():
            param.requires_grad = False

        return diffusion
    

    def forward(self, data, chunk_size: int = 64):
        x = data['clip'].to(self.device).float() # (B, C, T, H, W)
        b, c, f, h ,w = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)   # (B*T, C, H, W)

        if chunk_size is None or chunk_size >= x_flat.shape[0]:
            feat_flat = self.encoder(x_flat)                        # (B*T, C', H', W')
        else:
            feat_flat = torch.cat(
                [self.encoder(x_flat[i:i + chunk_size])
                 for i in range(0, x_flat.shape[0], chunk_size)],
                dim=0,
        )
        _, c2, h2, w2 = feat_flat.shape
        features_2d = feat_flat.view(b, f, c2, h2, w2).permute(0, 2, 1, 3, 4)
        features_2d = features_2d.contiguous()                      # (B, C', T, H', W')
        features    = self.head(features_2d)

        if self.use_demo:
            if not self.ppt_only:
                sex = data['demo']['sex'].to(self.device).long()
                sex_feature = self.sex_embedding_layer(sex).squeeze(1)
                features = torch.cat((features, sex_feature), dim=1)
            
                age = data['demo']['age'].to(self.device).float()
                age_feature = self.age_embedding_layer(age)
    
                features = torch.cat((features, age_feature), dim=1)
            
            ppt = data['demo']['ppt'].to(self.device).float()
            if ppt.ndim != 2 or ppt.shape[1] != 2:
                raise ValueError("demo['ppt'] must have shape (batch, 2), ordered pre, post")
            if self.intervention_matched_ppt:
                interventions = [str(v).strip().lower() for v in data['intervention']]
                if len(interventions) != ppt.shape[0] or any(v not in ('pre', 'post') for v in interventions):
                    raise ValueError("Each clip requires a Pre or Post intervention")
                mask = torch.tensor([[v == 'pre', v == 'post'] for v in interventions],
                                    device=ppt.device, dtype=torch.bool)
                # Bias-free embeddings make the unselected branch exactly zero.
                ppt = torch.where(mask, ppt, torch.zeros_like(ppt))
            ppt_pre_feature = self.ppt_pre_embedding_layer(ppt[:, 0:1])
            ppt_post_feature = self.ppt_post_embedding_layer(ppt[:, 1:2])
            features = torch.cat((features, ppt_pre_feature, ppt_post_feature), dim=1)


        out = self.last(features)
        return out
    
