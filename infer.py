import torch, numpy as np, torch.nn.functional as F
from model import UNet

def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and "state" in ck:
        net = UNet(residual=ck.get("residual", True)); net.load_state_dict(ck["state"])
    else:  # old best.pt (sigmoid model)
        net = UNet(residual=False); net.load_state_dict(ck)
    return net.to(device).eval()

@torch.no_grad()
def enhance_rgb(net, rgb, max_side=1024):
    """uint8 HxWx3 RGB -> enhanced uint8, same size, no 256x256 squeeze."""
    dev = next(net.parameters()).device
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    s = min(1.0, max_side / max(h, w))
    if s < 1: x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
    H, W = x.shape[2:]
    x = F.pad(x, (0, (-W) % 8, 0, (-H) % 8), mode="reflect")
    y = net(x)[:, :, :H, :W]
    if s < 1: y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
