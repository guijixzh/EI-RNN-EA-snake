import torch
import torch.nn as nn


# ==========================================
# CNN 前端编码器（test5b）
# ==========================================
# 输入：10x10x5 网格（channel_last） → 转置为 (B, 5, 10, 10)
# 结构：3 层小卷积 + GAP -> 32 维特征 -> 可学习投影头 32->24
#
# 关键设计：投影头输出保持 24 维，与 test5a 的 W_in (256x24) 完全兼容，
# 使 test5a_best_model.pth 可直接作为 EI 进化种子注入。
# ==========================================


class CNNEncoder(nn.Module):
    """轻量 CNN 编码器：网格 -> 32 维语义特征 -> 24 维 EI 输入。"""

    def __init__(self, in_channels=5, feat_dim=32, proj_dim=24):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim

        # 3 层小卷积，10x10 逐渐降维到 4x4，再 GAP
        # 参数量估算：5*16*9 + 16*32*9 + 32*32*9 ≈ 720 + 4608 + 9216 ≈ 1.45 万
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # GAP -> 32 维
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 投影头：32 -> 24（EI 输入维度，兼容 test5a）
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.Tanh(),
        )

        # 独立的动作分类头（仅预训练用）：32 -> 3
        # 训练监督信号作用在此头，避免污染投影头的语义；
        # 投影头只负责把任务语义线性映射到 EI 输入空间。
        self.action_head = nn.Linear(feat_dim, 3)

    def forward(self, grid):
        """grid: (B, 10, 10, 5) channel_last -> 返回 (B, proj_dim)。

        返回的 24 维向量直接作为 EI 网络的 obs_in。
        """
        feat = self.extract_feature(grid)
        out = self.proj(feat)                       # (B, 24)
        return out

    def forward_train(self, grid):
        """预训练专用：返回 (action_logits, proj_out)。

        action_logits: (B, 3) 动作分类 logits
        proj_out:      (B, 24) EI 输入投影
        """
        feat = self.extract_feature(grid)
        return self.action_head(feat), self.proj(feat)

    def extract_feature(self, grid):
        """返回 32 维语义特征（供可视化/分析）。"""
        x = grid.permute(0, 3, 1, 2).contiguous()
        x = self.convs(x)
        x = self.gap(x)
        return self.flatten(x)


def load_cnn_encoder(path, map_location='cpu'):
    """加载预训练 CNN 编码器（含投影头）。"""
    cnn = CNNEncoder()
    state = torch.load(path, map_location=map_location, weights_only=False)
    if 'model_state' in state:
        cnn.load_state_dict(state['model_state'])
    else:
        cnn.load_state_dict(state)
    cnn.eval()
    return cnn