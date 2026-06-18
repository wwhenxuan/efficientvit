# modified from https://github.com/facebookresearch/DiT
import math
from dataclasses import dataclass
from typing import Optional

import diffusers
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from efficientvit.diffusioncore.models.dit_sampler import create_diffusion
from efficientvit.models.utils.network import get_device

__all__ = ["DiTConfig", "DiT", "dc_ae_dit_xl_in_512px"]


@dataclass
class DiTConfig:
    """DiT 模型的全部超参数配置。"""

    name: str = "DiT"

    # --- 输入 / 网络结构 ---
    input_size: int = 32  # 输入图像（或 latent）的空间边长，如 32 表示 32×32
    patch_size: int = 2  # 每个 patch 的边长；patch 越大，token 数越少
    in_channels: int = 4  # 输入通道数；在 latent 扩散中通常是 VAE latent 的通道数（如 4）
    hidden_size: int = 1152  # Transformer 隐藏维度 D
    depth: int = 28  # DiTBlock 堆叠层数
    num_heads: int = 16  # 多头自注意力的头数
    mlp_ratio: float = 4.0  # FFN 中间层维度 = hidden_size * mlp_ratio
    post_norm: bool = False  # False=adaLN-Zero（DiT 默认）；True=post-norm 变体

    # --- 条件生成（类别引导）---
    class_dropout_prob: float = 0.1  # 训练时随机丢弃类别标签的概率，用于 Classifier-Free Guidance (CFG)
    num_classes: int = 1000  # 类别数（如 ImageNet 1000 类）
    learn_sigma: bool = True  # 是否同时预测噪声方差 σ；True 时输出通道数翻倍
    unconditional: bool = False  # True=无条件生成，不使用类别嵌入

    use_checkpoint: bool = True  # 是否对 DiTBlock 使用 gradient checkpointing 以节省显存

    # --- 预训练权重 ---
    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"  # 权重来源："dit" 或 "dc-ae"

    # --- 扩散调度器 ---
    eval_scheduler: str = "GaussianDiffusion"  # 推理调度器："GaussianDiffusion" 或 "UniPC"
    num_inference_steps: int = 250  # 推理步数（UniPC 使用）
    train_scheduler: str = "GaussianDiffusion"  # 训练调度器


def modulate(x, shift, scale, base: float = 1):
    """
    adaLN（Adaptive Layer Norm）调制函数，DiT 的核心条件注入机制。

    与标准 LayerNorm 不同，DiT 的 scale/shift 不是可学习参数，
    而是由条件向量 c（时间步 + 类别）通过 MLP 动态生成。
    公式：x' = x * (base + scale) + shift

    Args:
        x: (N, T, D) token 序列
        shift, scale: (N, D) 由条件 c 预测出的偏移和缩放
        base: 缩放基准，adaLN-Zero 用 1，post-norm 变体用 0
    """
    return x * (base + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                         2D 正弦-余弦位置编码工具函数                            #
#  DiT 不使用可学习的位置 embedding，而是用固定的 sin-cos 编码（与 ViT 相同）。      #
#################################################################################


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    生成 2D 网格的正弦-余弦位置编码，分别对 H 和 W 方向各编码 embed_dim/2 维后拼接。

    Args:
        grid_size: patch 网格边长，如 input_size=32, patch_size=2 → grid_size=16
    Returns:
        pos_embed: (grid_size², embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入层：将标量扩散时间步 t 编码为 D 维向量。

    扩散模型在每个时间步 t 都需要知道"当前噪声程度"。
    流程：t → 正弦/余弦位置编码 → 两层 MLP → 条件向量的一部分。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # 先将 t 映射到 frequency_embedding_size 维的正弦编码，再 MLP 投影到 hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成正弦/余弦时间步嵌入（与 Transformer 位置编码同族，来自 GLIDE/DiT 论文）。

        :param t: (N,) 每个样本的扩散时间步，可以是整数或小数
        :param dim: 输出嵌入维度
        :param max_period: 控制最低频率，越大则编码对 t 的变化越平滑
        :return: (N, dim) 时间步嵌入
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # (N, hidden_size)


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入层：将整数类别 ID 映射为 D 维向量，并支持 CFG 训练。

    Classifier-Free Guidance (CFG) 需要在推理时同时跑"有条件"和"无条件"两次前向。
    训练时通过 class_dropout_prob 随机把标签替换为"空标签"（num_classes 索引），
    让模型学会在无类别条件下也能去噪。
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # 多一行 embedding 给"空标签"（CFG 的无条件分支）
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        随机丢弃类别标签，用于 CFG 训练。
        被丢弃的样本标签会被设为 num_classes（空标签 embedding）。
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    DiT 的基本 Transformer 块，使用 adaLN-Zero 条件注入。

    结构类似 ViT block（自注意力 + FFN），但 LayerNorm 的仿射参数被移除，
    改由条件向量 c 通过 adaLN_modulation MLP 动态生成 scale/shift/gate。
    gate 初始化为 0（adaLN-Zero），使网络初始时近似恒等映射，训练更稳定。

    条件 c = t_emb + y_emb，贯穿所有 block，告诉模型"当前噪声水平"和"目标类别"。
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, post_norm=False, **block_kwargs):
        super().__init__()
        # elementwise_affine=False：不使用可学习的 γ/β，改由 adaLN 动态调制
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.post_norm = post_norm
        if not post_norm:
            # adaLN-Zero：为 MSA 和 MLP 各预测 shift/scale/gate，共 6 组参数
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        else:
            # post-norm 变体：无 gate，共 4 组参数
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))

    def forward(self, x: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        # x: (N, T, D) patch token 序列; c: (N, D) 条件向量
        # TODO: 这这里为DiT模型引入条件
        if not self.post_norm:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

            # Pre-norm + adaLN + gated residual
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        else:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            x = x + modulate(self.norm1(self.attn(x)), shift_msa, scale_msa, base=0)
            x = x + modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp, base=0)

        return x


class FinalLayer(nn.Module):
    """
    DiT 输出层：将 Transformer 隐状态投影回 patch 像素/latent 空间。

    每个 token 被线性映射为 patch_size² × out_channels 维，
    之后由 unpatchify 重组为 (N, C, H, W) 图像/latent。
    同样使用 adaLN 注入条件 c。
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 每个 token → 一个 patch 的所有像素值
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (N, T, patch_size² * out_channels)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)：用 Transformer 作为去噪网络的扩散模型。

    整体流程（推理）：
        1. 从纯高斯噪声 x_T 出发
        2. 逐步去噪 T → 0，每步调用 DiT 预测噪声 ε（或 x_0）
        3. 调度器根据预测更新 x_{t-1}
        4. 最终得到干净样本

    整体流程（训练）：
        1. 对干净样本 x_0 加噪得到 x_t
        2. DiT 预测噪声，与真实噪声计算 MSE loss

    条件信息（时间步 t + 类别 y）通过 adaLN 注入每个 Transformer block。
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        # learn_sigma=True 时同时预测 ε 和 σ，输出通道翻倍
        self.out_channels = cfg.in_channels * 2 if cfg.learn_sigma else cfg.in_channels

        # --- 输入嵌入 ---
        # PatchEmbed：将 (N,C,H,W) 切分为 patch 并线性投影为 (N,T,D) token 序列
        self.x_embedder = PatchEmbed(cfg.input_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size, bias=True)
        # 时间步嵌入：t → (N, D)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        # 类别嵌入：y → (N, D)，无条件模型跳过
        if not cfg.unconditional:
            self.y_embedder = LabelEmbedder(cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob)
        num_patches = self.x_embedder.num_patches  # T = (input_size/patch_size)²
        # 固定的 2D 正弦-余弦位置编码，不参与梯度更新（与 ViT 相同思路）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.hidden_size), requires_grad=False)

        # --- Transformer 主干 ---
        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.hidden_size, cfg.num_heads, mlp_ratio=cfg.mlp_ratio, post_norm=cfg.post_norm)
                for _ in range(cfg.depth)
            ]
        )
        # 输出层：token → patch 像素
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)
        if cfg.pretrained_path is not None:
            self.load_model()
        else:
            self.initialize_weights()

        # --- 扩散调度器（负责加噪/去噪的数学公式，与网络结构无关）---
        # eval_scheduler：推理时使用，逐步从噪声还原样本
        if cfg.eval_scheduler == "GaussianDiffusion":
            self.eval_scheduler = create_diffusion(str(250))  # 250 步 DDPM/DDIM 采样
        elif cfg.eval_scheduler == "UniPC":
            # UniPC：更少步数的高阶 ODE 求解器，通常 20~50 步即可
            self.eval_scheduler = diffusers.UniPCMultistepScheduler(
                solver_order=3,
                # rescale_betas_zero_snr=False,
                prediction_type="epsilon",  # 模型预测噪声 ε
            )
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

        # train_scheduler：训练时使用，负责随机采样 t 并计算 loss
        if cfg.train_scheduler == "GaussianDiffusion":
            self.train_scheduler = create_diffusion(timestep_respacing="")  # 完整 1000 步训练
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

    def get_trainable_modules(self) -> nn.ModuleDict:
        """返回需要训练/保存的模块字典，供训练框架统一调用。"""
        return nn.ModuleDict({"dit": self})

    def load_model(self):
        """从 checkpoint 加载预训练权重，支持 DiT 原版和 dc-ae 两种格式。"""
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "dit":
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            self.load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-ae":
            checkpoint = list(checkpoint["ema"].values())[0]
            self.get_trainable_modules().load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def initialize_weights(self):
        """
        权重初始化策略（DiT 论文 Section 4 及附录）。

        关键设计：adaLN 的最后一层和 final_layer 的 linear 初始化为 0，
        使模型初始输出 ≈ 0，等价于预测"无噪声"，训练初期 loss 较小、收敛更稳。
        """

        # 通用 Linear 层：Xavier 均匀初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 位置编码：用 2D sin-cos 填充，冻结不训练
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch 投影层：按 Linear 方式初始化（Conv2d 权重 reshape 后 Xavier）
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 类别 embedding：小方差正态分布
        if not self.cfg.unconditional:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # 时间步 MLP：小方差正态分布
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN 调制层最后一层置零 → gate/scale/shift 初始为 0，实现 adaLN-Zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 输出层同样置零 → 初始预测全 0
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        将 patch token 序列重组为空间图像/latent。

        PatchEmbed 的逆操作：
            (N, T, patch_size² * C) → (N, C, H, W)

        Args:
            x: (N, T, patch_size**2 * C)，T = (H/p) * (W/p)
        Returns:
            imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        """Gradient checkpointing 包装器：以计算换显存，反向传播时重新计算前向。"""

        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward_without_cfg(self, x, t, y):
        """
        DiT 核心前向传播（不含 Classifier-Free Guidance）。

        Args:
            x: (N, C, H, W) 带噪输入（训练时是 x_t，推理时是当前步的样本）
            t: (N,) 扩散时间步
            y: (N,) 类别标签（整数 ID）
        Returns:
            (N, out_channels, H, W) 模型预测（噪声 ε，learn_sigma 时还含 σ）
        """
        # Step 1: 图像/latent → patch token + 位置编码
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), T = H*W / patch_size²
        # Step 2: 构建条件向量 c = 时间步嵌入 + 类别嵌入
        t = self.t_embedder(t)  # (N, D)
        if self.cfg.unconditional:
            c = t
        else:
            y = self.y_embedder(y, self.training)  # (N, D)，训练时可能随机 drop 标签
            c = t + y  # (N, D)
        # Step 3: 逐层 Transformer block，c 通过 adaLN 注入每一层
        for block in self.blocks:
            if self.cfg.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)  # (N, T, D)
            else:
                x = block(x, c)
        # Step 4: 输出层 + unpatchify 还原空间维度
        x = self.final_layer(x, c)  # (N, T, patch_size² * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        带 Classifier-Free Guidance (CFG) 的前向传播。

        CFG 公式：ε_cfg = ε_uncond + scale * (ε_cond - ε_uncond)
        通过 batch 拼接实现：前半 batch 用真实标签，后半 batch 用空标签，
        一次前向同时得到有条件和无条件预测，再线性组合。

        Args:
            x: (2N, C, H, W) 已拼接的噪声样本（前半和后半相同）
            t: (2N,) 时间步
            y: (2N,) 标签（前半真实类，后半空类）
            cfg_scale: CFG 强度，越大类别条件越强、多样性越低
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)  # 复制一份，配合 y 的前半/后半标签
        model_out = self.forward_without_cfg(combined, t, y)
        # learn_sigma 时输出分两半：前半是 ε，后半是 σ（方差）
        eps, rest = model_out[:, : self.cfg.in_channels], model_out[:, self.cfg.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, y, generator: Optional[torch.Generator] = None):
        """
        训练入口：随机采样时间步 t，对 x_0 加噪，计算去噪 MSE loss。

        Args:
            x: (N, C, H, W) 干净样本 x_0（或 VAE latent）
            y: (N,) 类别标签
        Returns:
            loss: 标量 MSE loss
            info: 包含 loss_dict 的字典，供日志记录
        """
        info = {}
        device = x.device
        if self.cfg.train_scheduler == "GaussianDiffusion":
            model_kwargs = dict(y=y)
            # 每个样本随机采样一个 t ∈ [0, num_timesteps)
            timesteps = torch.randint(0, self.train_scheduler.num_timesteps, (x.shape[0],), device=device)
            # train_scheduler 内部：加噪 x_t → 调用 model → 计算 MSE(预测噪声, 真实噪声)
            loss_dict = self.train_scheduler.training_losses(self.forward_without_cfg, x, timesteps, model_kwargs)
            loss = loss_dict["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")
        info["loss_dict"] = {"loss": loss}
        return loss, info

    @torch.no_grad()
    def generate(
        self, inputs, null_inputs, scale: float = 1.5, generator: Optional[torch.Generator] = None, progress=False
    ):
        """
        推理入口：从纯噪声逐步去噪，生成样本。

        Args:
            inputs: (N,) 目标类别标签
            null_inputs: (N,) 空标签（CFG 无条件分支用），scale=1.0 时可忽略
            scale: CFG 强度，1.0=不使用 CFG，>1 增强类别条件
            generator: 随机数生成器，控制初始噪声的可复现性
            progress: 是否显示采样进度条
        Returns:
            samples: (N, C, H, W) 生成的 latent 或图像
        """
        device = get_device(self)
        # 从标准高斯噪声 x_T 开始
        samples = torch.randn(
            (inputs.shape[0], self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size),
            generator=generator,
            device=device,
        )

        if scale != 1.0:
            # --- 使用 CFG：batch 翻倍，同时跑有条件和无条件 ---
            assert null_inputs is not None
            samples = torch.cat([samples, samples], dim=0)  # (2N, C, H, W)
            inputs = torch.cat([inputs, null_inputs], dim=0)  # 前半真实类，后半空类
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs, cfg_scale=scale)
                # p_sample_loop：从 t=T 迭代到 t=0，每步调用 forward_with_cfg
                samples = self.eval_scheduler.p_sample_loop(
                    self.forward_with_cfg,
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=progress,
                    device=device,
                )
            elif self.cfg.eval_scheduler == "UniPC":
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_with_cfg(samples, timesteps, inputs, scale)
                    if self.cfg.learn_sigma:
                        model_output = model_output[:, : self.cfg.in_channels]  # 只用 ε，忽略 σ
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")
            # 去掉 CFG 复制的后半 batch，只保留有条件分支的结果
            samples, _ = samples.chunk(2, dim=0)
        else:
            # --- 不使用 CFG：单次前向 ---
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs)
                samples = self.eval_scheduler.p_sample_loop(
                    self.forward_without_cfg,
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=progress,
                    device=device,
                )
            elif self.cfg.eval_scheduler == "UniPC":
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_without_cfg(samples, timesteps, inputs)
                    if self.cfg.learn_sigma:
                        model_output = model_output[:, : self.cfg.in_channels]
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")

        return samples


def dc_ae_dit_xl_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=dit dit.depth=28 dit.hidden_size=1152 dit.num_heads=16 dit.in_channels={in_channels} dit.patch_size=1 "
        f"dit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )
