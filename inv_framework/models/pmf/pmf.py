import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pmfDiT
from .pmf_controlnet import PMFControlNet


def _optional_msssim():
    try:
        from pytorch_msssim import ms_ssim
    except ImportError as error:
        raise ImportError(
            "MS-SSIM training loss requires the optional 'train' dependencies"
        ) from error
    return ms_ssim


def _optional_lpips():
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "LPIPS training loss requires the optional 'train' dependencies"
        ) from error
    return lpips


def paired_random_resized_crop(x1, x2, out_size=224,
                               scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    """Apply identical random crop+resize to two paired tensors.

    x1, x2: (B, C, H, W). Output: (B, C, out_size, out_size).
    """
    assert x1.shape == x2.shape
    B, _, H, W = x1.shape
    device, dtype = x1.device, x1.dtype

    target_area = (
        torch.empty(B, device=device).uniform_(scale[0], scale[1]) * (H * W)
    )
    log_ratio = torch.tensor(ratio, device=device).log()
    aspect = torch.exp(
        torch.empty(B, device=device).uniform_(log_ratio[0].item(), log_ratio[1].item())
    )

    crop_w = torch.clamp(torch.round(torch.sqrt(target_area * aspect)).long(), 1, W)
    crop_h = torch.clamp(torch.round(torch.sqrt(target_area / aspect)).long(), 1, H)
    top  = (torch.rand(B, device=device) * (H - crop_h + 1).float()).long()
    left = (torch.rand(B, device=device) * (W - crop_w + 1).float()).long()

    a_x = crop_w.float() / W
    a_y = crop_h.float() / H
    b_x = (2.0 * left.float() + crop_w.float() - W) / W
    b_y = (2.0 * top.float()  + crop_h.float() - H) / H

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = a_x
    theta[:, 1, 1] = a_y
    theta[:, 0, 2] = b_x
    theta[:, 1, 2] = b_y

    grid = F.affine_grid(theta, (B, x1.shape[1], out_size, out_size),
                         align_corners=False)
    y1 = F.grid_sample(x1, grid, mode="bilinear",
                       padding_mode="border", align_corners=False)
    y2 = F.grid_sample(x2, grid, mode="bilinear",
                       padding_mode="border", align_corners=False)
    return y1, y2

class pixelMeanFlow(nn.Module):
    """pixel MeanFlow"""

    _noise_scale_table = {
        'pmfDiT_B_16': 1.0,
        'pmfDiT_B_32': 2.0,
        'pmfDiT_B_64': 4.0,
        'pmfDiT_B_128': 8.0,
        'pmfDiT_L_16': 1.0,
        'pmfDiT_L_32': 4.0,
        'pmfDiT_H_16': 2.0,
        'pmfDiT_H_32': 4.0,
    }

    def __init__(
        self,
        model_str: str,
        dtype: torch.dtype = torch.float32,
        img_size: int = 256,
        img_channels: int = 3,
        num_classes: int = 1000,
        eval: bool = False,
        # Training hyperparameters
        P_mean: float = -0.4,
        P_std: float = 1.0,
        cfg_max: float = 7.0,
        noise_scale: float = None,
        data_proportion: float = 0.5,
        cfg_beta: float = 1.0,
        class_dropout_prob: float = 0.1,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        tr_uniform: bool = False,
        # MS-SSIM auxiliary loss
        msssim_lambda: float = 0.0,
        msssim_max_t: float = 0.5,
        # LPIPS auxiliary loss
        lpips_lambda: float = 0.0,
        lpips_net: str = "vgg",
        lpips_crop_size: int = 224,
        perceptual_max_t: float = 0.8,
    ):
        super().__init__()
        self.model_str = model_str
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Training hyperparameters
        self.P_mean = P_mean
        self.P_std = P_std
        self.cfg_max = cfg_max
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.tr_uniform = tr_uniform
        self.msssim_lambda = msssim_lambda
        self.msssim_max_t = msssim_max_t

        self.lpips_lambda = lpips_lambda
        self.lpips_crop_size = lpips_crop_size
        self.perceptual_max_t = perceptual_max_t

        if lpips_lambda > 0:
            self.lpips_model = _optional_lpips().LPIPS(net=lpips_net, verbose=False)
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

        net_fn = getattr(pmfDiT, self.model_str)
        self.net: pmfDiT.pmfDiT = net_fn(
            input_size=self.img_size,
            in_channels=self.img_channels,
            num_classes=self.num_classes,
            eval_mode=eval
        )

        # noise scaling: use provided value or look up from table
        if noise_scale is not None:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = self._noise_scale_table[model_str]

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz, device):
        """Sample from logit-normal distribution."""
        rnd_normal = torch.randn(bz, 1, 1, 1, dtype=self.dtype, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device):
        """
        Sample t and r from logit-normal distribution.
        Returns: (t, r, fm_mask) where fm_mask marks flow matching samples (t==r).
        """
        t = self.logit_normal_dist(bz, device)
        r = self.logit_normal_dist(bz, device)

        if self.tr_uniform:
            # 10% random uniform tr samples
            unif_mask = torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device) < 0.1
            t = torch.where(unif_mask, torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device), t)
            r = torch.where(unif_mask, torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device), r)

        data_size = int(bz * self.data_proportion)
        fm_mask = torch.arange(bz, device=device) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, device, s_max=7.0):
        """Sample CFG scale omega from power distribution."""
        u = torch.rand(bz, 1, 1, 1, dtype=torch.float32, device=device)
        s_max_t = torch.tensor(s_max, dtype=torch.float32, device=device)

        if self.cfg_beta == 1.0:  # special case for \int 1/x
            s = torch.exp(u * torch.log1p(s_max_t))
        else:
            b = torch.tensor(self.cfg_beta, dtype=torch.float32, device=device)
            log_base = (1.0 - b) * torch.log1p(s_max_t)
            log_inner = torch.log1p(u * torch.expm1(log_base))
            s = torch.exp(log_inner / (1.0 - b))

        return s.to(torch.float32)

    def sample_cfg_interval(self, bz, device, fm_mask=None):
        """Sample CFG interval [t_min, t_max] from uniform distribution."""
        t_min = torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device) * 0.5
        t_max = torch.rand(bz, 1, 1, 1, dtype=self.dtype, device=device) * 0.5 + 0.5

        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        """
        Compute the predicted v component conditioned on class labels.
        Uses h=0, t_min=0, t_max=1 as dummy values.
        """
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]
        return v

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned predicted v components.
        Doubles the batch: first half conditioned, second half unconditioned.
        """
        bz = x.shape[0]
        device = x.device

        x = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, dtype=y.dtype, device=device)
        y = torch.cat([y, y_null], dim=0)
        t = torch.cat([t, t], dim=0)
        w = torch.cat([omega, torch.ones_like(omega)], dim=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = torch.chunk(out, 2, dim=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG training.
        For dropped samples, v_g is replaced with v_t (unguided).
        """
        bz = v_t.shape[0]
        device = v_t.device

        rand_mask = torch.rand(bz, device=device) < self.class_dropout_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=device)[:, None, None, None] < num_drop

        labels = torch.where(
            drop_mask.reshape(bz),
            torch.tensor(self.num_classes, dtype=labels.dtype, device=device),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Returns:
            v_g: Guided instantaneous velocity target for training.
            v_c: Conditioned instantaneous velocity for jvp computation.
        """
        # compute CFG target
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w_interval = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))

        v_c_interval = self.v_cond_fn(z_t, t, w_interval, y=y)
        v_g = v_t + (1 - 1 / w_interval) * (v_c_interval - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = torch.where(fm_mask, v_g_fm, v_g)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, labels, noise=None):
        """
        Forward process of pixel MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, C, H, W).
            labels: Corresponding class labels, shape (B,).
            noise: Optional paired standard Gaussian noise, shape (B, C, H, W).

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.to(self.dtype)
        bz = images.shape[0]
        device = images.device

        # Sample t, r and flow matching mask
        t, r, fm_mask = self.sample_tr(bz, device)

        # Generate noise and noisy image
        if noise is None:
            e = torch.randn_like(x) * self.noise_scale
        else:
            e = noise.to(device=device, dtype=self.dtype) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(min=0.05)

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, device, fm_mask)
        omega = self.sample_cfg_scale(bz, device, s_max=self.cfg_max)

        # Compute guided velocity v_g and conditioned velocity v_c
        with torch.no_grad():
            v_g, v_c = self.guidance_fn(
                v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
            )

        # Cond dropout (dropout class labels)
        with torch.no_grad():
            labels, v_g = self.cond_drop(v_t, v_g, labels)

        # Regular forward pass for u, v (preserves autograd graph for model params)
        u, v = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

        # JVP only for du_dt (will be detached, so no need for param gradients)
        def u_fn_for_jvp(z_t_in, t_in, r_in):
            return self.u_fn(z_t_in, t_in, t_in - r_in, omega, t_min, t_max, y=labels)

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)

        _, (du_dt, _dv_dt) = torch.autograd.functional.jvp(
            u_fn_for_jvp,
            (z_t, t, r),
            (v_c, dtdt, dtdr),
        )

        # Compound function V = u + (t - r) * du/dt
        V = u + (t - r) * du_dt.detach()

        v_g = v_g.detach()

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / adp_wt.detach()

        # pixel MeanFlow objective: v-loss
        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss
        loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        # MS-SSIM auxiliary loss on predicted clean image
        if self.msssim_lambda > 0:
            with torch.no_grad():
                pred_x = z_t - t * u  # predicted clean image
                # normalize to [0, 1] for ms_ssim
                pred_x_01 = (pred_x.detach().float() + 1) / 2
                x_01 = (x.float() + 1) / 2
                pred_x_01 = pred_x_01.clamp(0, 1)
                x_01 = x_01.clamp(0, 1)

            # recompute pred_x with grad for loss backprop
            pred_x = z_t.detach() - t * u
            pred_x_01_grad = (pred_x.float() + 1) / 2
            pred_x_01_grad = pred_x_01_grad.clamp(0, 1)

            # ms_ssim returns a scalar in [0, 1], 1 = identical
            msssim_val = _optional_msssim()(
                pred_x_01_grad, x_01.detach(), data_range=1.0, size_average=False
            )
            # loss = 1 - ms_ssim, per sample
            loss_msssim = 1.0 - msssim_val

            # only apply when t < msssim_max_t (small t = cleaner prediction)
            mask = (t.flatten() < self.msssim_max_t).float()
            loss_msssim = loss_msssim * mask
            loss_msssim = adp_wt_fn(loss_msssim)
        else:
            loss_msssim = torch.zeros(bz, device=device)

        # LPIPS auxiliary loss on predicted clean image (with paired random
        # resized crop to break exact pixel correspondence)
        if self.lpips_lambda > 0:
            pred_x_lpips = z_t.detach() - t * u  # in [-1, 1] approx
            x_target = x.detach()

            # paired random resized crop to lpips_crop_size
            pred_crop, x_crop = paired_random_resized_crop(
                pred_x_lpips, x_target, out_size=self.lpips_crop_size
            )

            # grayscale -> 3-channel by repeat (LPIPS expects RGB in [-1, 1])
            if pred_crop.shape[1] == 1:
                pred_crop = pred_crop.repeat(1, 3, 1, 1)
                x_crop = x_crop.repeat(1, 3, 1, 1)

            # LPIPS expects float32 inputs in [-1, 1]; clamp for safety
            pred_crop = pred_crop.float().clamp(-1.0, 1.0)
            x_crop    = x_crop.float().clamp(-1.0, 1.0)

            lpips_dist = self.lpips_model(pred_crop, x_crop).reshape(bz)

            mask = (t.flatten() < self.perceptual_max_t).float()
            loss_lpips = lpips_dist * mask
            loss_lpips = adp_wt_fn(loss_lpips)
        else:
            loss_lpips = torch.zeros(bz, device=device)

        loss = (loss_u + loss_v
                + self.msssim_lambda * loss_msssim
                + self.lpips_lambda * loss_lpips).mean()

        dict_losses = {
            "loss": loss.detach(),
            "loss_u": torch.mean((V - v_g) ** 2).detach(),
            "loss_v": torch.mean((v - v_g) ** 2).detach(),
            "loss_msssim": loss_msssim.mean().detach(),
            "loss_lpips": loss_lpips.mean().detach(),
        }

        return loss, dict_losses

    #######################################################
    #                    Inference                        #
    #######################################################

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
        """
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]

        t = t.expand(bsz)
        r = r.expand(bsz)
        omega = omega.expand(bsz)
        t_min = t_min.expand(bsz)
        t_max = t_max.expand(bsz)

        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]

        return z_t - (t - r)[:, None, None, None] * u

    @torch.no_grad()
    def generate(self, n_sample, rng, num_steps, omega, t_min, t_max,
                 labels=None):
        """
        Generate samples from the model

        Args:
            n_sample: Number of samples to generate.
            num_steps: Number of sampling steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            labels: Optional labels (if not provided, random labels will be used).
        """
        x_shape = (n_sample, self.img_channels, self.img_size, self.img_size)
        z_t = rng.randn(x_shape).to(self.dtype) * self.noise_scale

        if labels is not None:
            y = labels.to(z_t.device)
        else:
            y = rng.randint(0, self.num_classes, size=(n_sample,), dtype=torch.int32).to(z_t.device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(self.dtype).to(z_t.device)

        omega = torch.tensor(omega, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(omega) else omega
        t_min = torch.tensor(t_min, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(t_min) else t_min
        t_max = torch.tensor(t_max, dtype=self.dtype, device=z_t.device) if not torch.is_tensor(t_max) else t_max

        for i in range(num_steps):
            # the main inference algorithm here
            t = t_steps[i]
            r = t_steps[i + 1]
            bsz = z_t.shape[0]
            t_b = t.expand(bsz)
            r_b = r.expand(bsz)
            omega_b = omega.expand(bsz)
            t_min_b = t_min.expand(bsz)
            t_max_b = t_max.expand(bsz)

            u = self.u_fn(z_t, t_b, t_b - r_b, omega_b, t_min_b, t_max_b, y=y)[0]
            z_t = z_t - (t_b - r_b)[:, None, None, None] * u

        return z_t


#######################################################
#               pMF + ControlNet wrapper              #
#######################################################


class pixelMeanFlowControlNet(nn.Module):
    """Holds a frozen pMF base + a trainable PMFControlNet (PixArt-alpha style).

    During training the base velocity target v_g is computed via base only
    (no CN, no_grad) so the CN doesn't appear on both sides of the loss --
    otherwise it would learn to match its own prediction, a degenerate target.

    The main forward pass that produces (u, v) with autograd graph routes
    through the CN: `u_fn(..., cond=cond)` interleaves shared_blocks with CN
    injections following the PixArt-alpha skip-first convention.
    """

    def __init__(self, base: pixelMeanFlow, copy_blocks_num: int = 4):
        super().__init__()
        self.base = base
        # Freeze every parameter in base (incl. lpips_model if any).
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.cn = PMFControlNet(base.net, copy_blocks_num=copy_blocks_num)
        self.copy_blocks_num = copy_blocks_num

    # ------------------------------------------------------------------
    # u_fn with optional cond. cond=None falls back to plain base.u_fn.
    # ------------------------------------------------------------------

    def _u_fn_with_cond(self, x, t, h, omega, t_min, t_max, y, cond):
        """Run base.net forward but inject CN c_skips into shared_blocks[1..N]."""
        net = self.base.net
        rope = net.rope_freqs
        bz = x.shape[0]

        h_1d = h.reshape(bz)
        omega_1d = omega.reshape(bz)
        t_min_1d = t_min.reshape(bz)
        t_max_1d = t_max.reshape(bz)

        # Build seq for z_t and cn_seq for cond, sharing the same prefix layout.
        seq = net._build_sequence(x, h_1d, omega_1d, t_min_1d, t_max_1d, y)
        cn_seq = net._build_sequence(cond, h_1d, omega_1d, t_min_1d, t_max_1d, y)

        # PixArt-alpha skip-first: shared_blocks[0] runs un-controlled.
        seq = net.shared_blocks[0](seq, rope)

        # CN forward -> list of c_skip residuals.
        controls = self.cn(seq, cn_seq, rope)

        # Inject c_skips into base.shared_blocks[1..N].
        N = len(controls)
        for i in range(N):
            seq = net.shared_blocks[i + 1](seq + controls[i], rope)
        # Remaining shared blocks run un-controlled.
        for block in net.shared_blocks[N + 1:]:
            seq = block(seq, rope)

        # u/v heads (frozen).
        u_seq = v_seq = seq
        for block in net.u_heads:
            u_seq = block(u_seq, rope)
        for block in net.v_heads:
            v_seq = block(v_seq, rope)

        u_tokens = u_seq[:, net.prefix_tokens:]
        v_tokens = v_seq[:, net.prefix_tokens:]
        u_pred = net.unpatchify(net.u_final_layer(u_tokens))
        v_pred = net.unpatchify(net.v_final_layer(v_tokens))

        # Reparametrize (matches base.net.forward).
        t_b = t.reshape(bz, 1, 1, 1)
        u = (x - u_pred) / torch.clamp(t_b, min=0.05)
        v = (x - v_pred) / torch.clamp(t_b, min=0.05)
        return u, v

    def u_fn(self, x, t, h, omega, t_min, t_max, y, cond=None):
        if cond is None:
            return self.base.u_fn(x, t, h, omega, t_min, t_max, y)
        return self._u_fn_with_cond(x, t, h, omega, t_min, t_max, y, cond)

    # ------------------------------------------------------------------
    # Training forward (mirrors base.forward but threads cond through).
    # ------------------------------------------------------------------

    def forward(self, images, conditions, labels, noise=None):
        x = images.to(self.base.dtype)
        bz = x.shape[0]
        device = x.device
        cond = conditions.to(device=device, dtype=self.base.dtype)

        # Sample t, r and FM mask (delegate to base helper).
        t, r, fm_mask = self.base.sample_tr(bz, device)

        if noise is None:
            e = torch.randn_like(x) * self.base.noise_scale
        else:
            e = noise.to(device=device, dtype=self.base.dtype) * self.base.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(min=0.05)

        t_min, t_max = self.base.sample_cfg_interval(bz, device, fm_mask)
        omega = self.base.sample_cfg_scale(bz, device, s_max=self.base.cfg_max)

        # v_g target via BASE ONLY (no CN, no cond) -- avoids degenerate self-target.
        with torch.no_grad():
            v_g, v_c = self.base.guidance_fn(
                v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
            )
        with torch.no_grad():
            labels, v_g = self.base.cond_drop(v_t, v_g, labels)

        # Main forward WITH CN and cond.
        u, v = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels, cond=cond)

        # JVP for du/dt. cond enters via closure as a constant.
        def u_fn_for_jvp(z_t_in, t_in, r_in):
            return self.u_fn(z_t_in, t_in, t_in - r_in, omega, t_min, t_max,
                             y=labels, cond=cond)

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)
        _, (du_dt, _dv_dt) = torch.autograd.functional.jvp(
            u_fn_for_jvp,
            (z_t, t, r),
            (v_c, dtdt, dtdr),
        )
        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()

        norm_p = self.base.norm_p
        norm_eps = self.base.norm_eps

        def adp_wt_fn(loss):
            adp_wt = (loss + norm_eps) ** norm_p
            return loss / adp_wt.detach()

        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)
        loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        # Aux losses (msssim/lpips) -- only active if base was built with them.
        if self.base.msssim_lambda > 0:
            with torch.no_grad():
                pred_x_det = z_t - t * u
                pred_x_01 = ((pred_x_det.detach().float() + 1) / 2).clamp(0, 1)
                x_01 = ((x.float() + 1) / 2).clamp(0, 1)
            pred_x = z_t.detach() - t * u
            pred_x_01_grad = ((pred_x.float() + 1) / 2).clamp(0, 1)
            msssim_val = _optional_msssim()(pred_x_01_grad, x_01.detach(),
                                  data_range=1.0, size_average=False)
            mask = (t.flatten() < self.base.msssim_max_t).float()
            loss_msssim = adp_wt_fn((1.0 - msssim_val) * mask)
        else:
            loss_msssim = torch.zeros(bz, device=device)

        if self.base.lpips_lambda > 0 and self.base.lpips_model is not None:
            pred_x_lpips = z_t.detach() - t * u
            x_target = x.detach()
            pred_crop, x_crop = paired_random_resized_crop(
                pred_x_lpips, x_target, out_size=self.base.lpips_crop_size
            )
            if pred_crop.shape[1] == 1:
                pred_crop = pred_crop.repeat(1, 3, 1, 1)
                x_crop = x_crop.repeat(1, 3, 1, 1)
            pred_crop = pred_crop.float().clamp(-1.0, 1.0)
            x_crop = x_crop.float().clamp(-1.0, 1.0)
            lpips_dist = self.base.lpips_model(pred_crop, x_crop).reshape(bz)
            mask = (t.flatten() < self.base.perceptual_max_t).float()
            loss_lpips = adp_wt_fn(lpips_dist * mask)
        else:
            loss_lpips = torch.zeros(bz, device=device)

        loss = (loss_u + loss_v
                + self.base.msssim_lambda * loss_msssim
                + self.base.lpips_lambda * loss_lpips).mean()

        dict_losses = {
            "loss": loss.detach(),
            "loss_u": torch.mean((V - v_g) ** 2).detach(),
            "loss_v": torch.mean((v - v_g) ** 2).detach(),
            "loss_msssim": loss_msssim.mean().detach(),
            "loss_lpips": loss_lpips.mean().detach(),
        }
        return loss, dict_losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, n_sample, rng, num_steps, omega, t_min, t_max,
                 conditions, labels=None):
        device = next(self.cn.parameters()).device
        cond = conditions.to(device=device, dtype=self.base.dtype)
        x_shape = (n_sample, self.base.img_channels, self.base.img_size, self.base.img_size)
        z_t = rng.randn(x_shape).to(self.base.dtype) * self.base.noise_scale

        if labels is not None:
            y = labels.to(z_t.device)
        else:
            y = rng.randint(0, self.base.num_classes, size=(n_sample,),
                            dtype=torch.int32).to(z_t.device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(self.base.dtype).to(z_t.device)
        omega_t = (torch.tensor(omega, dtype=self.base.dtype, device=z_t.device)
                   if not torch.is_tensor(omega) else omega)
        t_min_t = (torch.tensor(t_min, dtype=self.base.dtype, device=z_t.device)
                   if not torch.is_tensor(t_min) else t_min)
        t_max_t = (torch.tensor(t_max, dtype=self.base.dtype, device=z_t.device)
                   if not torch.is_tensor(t_max) else t_max)

        for i in range(num_steps):
            t = t_steps[i].expand(n_sample)
            r = t_steps[i + 1].expand(n_sample)
            u = self.u_fn(z_t, t, t - r,
                          omega_t.expand(n_sample),
                          t_min_t.expand(n_sample),
                          t_max_t.expand(n_sample),
                          y=y, cond=cond)[0]
            z_t = z_t - (t - r)[:, None, None, None] * u

        return z_t
