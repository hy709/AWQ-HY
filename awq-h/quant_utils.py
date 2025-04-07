import torch
from logger_config import logger

def pseudo_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False):
    """
    伪量化张量
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0, "张量的最后一维必须能被 q_group_size 整除"
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2, "输入张量必须是二维的（或通过分组量化变为二维）"

    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = torch.zeros_like(scales)  # 如果不使用零点偏移，零点为 0

    assert torch.isnan(scales).sum() == 0, "存在 NaN 的 scale 值"
    assert torch.isnan(w).sum() == 0, "输入张量中存在 NaN 值"

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
                    torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales

    assert torch.isnan(w).sum() == 0, "量化后的张量中存在 NaN 值"

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        # 确保 scales 和 zeros 形状正确，并处理潜在的维度不一致
        scale_reshape = scales.reshape(-1) if scales.numel() == w.shape[0] else scales.view(w.shape[0], -1)
        zero_reshape = zeros.reshape(-1) if zeros.numel() == w.shape[0] else zeros.view(w.shape[0], -1)
        return w, scale_reshape, zero_reshape
    else:
        return w
    
def w_quantize_func(p, n_bit=8, zero_point=True):
    """
    使用伪量化函数，支持自定义参数
    """
    return pseudo_quantize_tensor(
        p, n_bit=n_bit, zero_point=zero_point
    ).detach()