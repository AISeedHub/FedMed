"""
SegResNet with Cirrhosis Classification (FedMorph model)
=========================================================

Architecture:
  MONAI SegResNet (init_filters=8, ~1M params)
  + MorphologicalDescriptor: differentiable vol_ratios / centroids / compactness
  + SegmentFeatureFusion: bottleneck features + morph features → cirrhosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet


class MorphologicalDescriptor(nn.Module):
    """Differentiable morphological features from predicted segment masks.

    Outputs (morph_vector, vol_ratios):
      morph_vector = [vol_ratios | centroids_flat | compactness]  dim = 5*C
      vol_ratios   = per-segment volume ratios                    dim = C
    """

    def __init__(self, num_segments: int = 9):
        super().__init__()
        self.num_segments = num_segments

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, seg_logits):
        seg_logits = seg_logits.float()
        seg_probs = torch.sigmoid(seg_logits)
        B, C, D, H, W = seg_probs.shape
        device = seg_probs.device

        volumes = seg_probs.sum(dim=(2, 3, 4))
        total_vol = volumes.sum(dim=1, keepdim=True).clamp(min=1.0)
        vol_ratios = volumes / total_vol

        grid_d = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        grid_h = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        grid_w = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
        vol_safe = volumes.clamp(min=1.0)
        cent_d = (seg_probs * grid_d).sum(dim=(2, 3, 4)) / vol_safe
        cent_h = (seg_probs * grid_h).sum(dim=(2, 3, 4)) / vol_safe
        cent_w = (seg_probs * grid_w).sum(dim=(2, 3, 4)) / vol_safe
        centroids = torch.stack([cent_d, cent_h, cent_w], dim=2)

        dx = (seg_probs[:, :, 1:] - seg_probs[:, :, :-1]).abs().sum(dim=(2, 3, 4))
        dy = (seg_probs[:, :, :, 1:] - seg_probs[:, :, :, :-1]).abs().sum(
            dim=(2, 3, 4)
        )
        dz = (seg_probs[:, :, :, :, 1:] - seg_probs[:, :, :, :, :-1]).abs().sum(
            dim=(2, 3, 4)
        )
        surface = dx + dy + dz
        compactness = surface / vol_safe.pow(2.0 / 3.0).clamp(min=1e-3)
        comp_max = compactness.max(dim=1, keepdim=True)[0].clamp(min=1e-3)
        compactness = compactness / comp_max

        morph = torch.cat([vol_ratios, centroids.flatten(1), compactness], dim=1)
        morph = torch.nan_to_num(morph, nan=0.0, posinf=1.0, neginf=0.0)
        vol_ratios = torch.nan_to_num(vol_ratios, nan=0.0)
        return morph, vol_ratios


class SegmentFeatureFusion(nn.Module):
    """9-segment masked feature fusion + morph features → cirrhosis logit."""

    def __init__(
        self,
        in_channels: int,
        num_segments: int = 9,
        hidden_dim: int = 32,
        morph_dim: int | None = None,
    ):
        super().__init__()
        if morph_dim is None:
            morph_dim = num_segments + num_segments * 3 + num_segments
        self.num_segments = num_segments
        self.seg_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_segments + 1), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, features, seg_logits, morph_feats):
        features = features.float()
        seg_logits = seg_logits.float()
        morph_feats = morph_feats.float()
        seg_probs = torch.sigmoid(seg_logits)
        if seg_probs.shape[2:] != features.shape[2:]:
            seg_probs = F.interpolate(
                seg_probs,
                size=features.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        seg_feats = []
        for i in range(self.num_segments):
            mask_i = seg_probs[:, i : i + 1]
            masked = features * mask_i
            denom = mask_i.sum(dim=(2, 3, 4)).clamp(min=1.0)
            pooled = masked.sum(dim=(2, 3, 4)) / denom
            seg_feats.append(self.seg_proj(pooled))

        fused_seg = torch.cat(seg_feats, dim=1)
        morph_proj = self.morph_proj(morph_feats)
        fused = torch.cat([fused_seg, morph_proj], dim=1)
        return self.classifier(fused)


class SegResNetWithCirrhosis(SegResNet):
    """SegResNet + MorphologicalDescriptor + SegmentFeatureFusion.

    Returns: (seg_logits, cls_logits, morph_feats, vol_ratios)
    """

    def __init__(self, num_segments: int = 9, cls_hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        bottleneck_ch = self.init_filters * (2 ** (len(self.blocks_down) - 1))
        self.morph_desc = MorphologicalDescriptor(num_segments)
        self.cls_head = SegmentFeatureFusion(
            in_channels=bottleneck_ch,
            num_segments=num_segments,
            hidden_dim=cls_hidden,
        )

    def forward(self, x):
        x_enc, down_x = self.encode(x)
        down_x.reverse()
        seg_logits = self.decode(x_enc, down_x)
        morph_feats, vol_ratios = self.morph_desc(seg_logits)
        cls_logits = self.cls_head(x_enc, seg_logits, morph_feats)
        return seg_logits, cls_logits, morph_feats, vol_ratios


def build_model(config: dict, device: torch.device) -> SegResNetWithCirrhosis:
    """Build SegResNetWithCirrhosis from config dict."""
    num_classes = config["num_classes"]
    init_filters = config["init_filters"]
    blocks_down = tuple(config["blocks_down"])
    blocks_up = tuple(config["blocks_up"])

    model = SegResNetWithCirrhosis(
        num_segments=num_classes,
        cls_hidden=32,
        spatial_dims=3,
        init_filters=init_filters,
        in_channels=1,
        out_channels=num_classes,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        norm=("GROUP", {"num_groups": min(8, init_filters)}),
        dropout_prob=None,
        upsample_mode="nontrainable",
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters())
    print(
        f"  SegResNet params: {n_p:,} "
        f"(init_filters={init_filters}, blocks_down={blocks_down})"
    )
    return model
