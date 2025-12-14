import math
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def test_patch_order(img, patches, patch_size):
    C, H, W = img.shape
    ph, pw = patch_size
    gh = H // ph
    gw = W // pw

    print(torch.allclose(patches[0], img[:, 0:ph, 0:pw]))
    print(torch.allclose(patches[1], img[:, 0:ph, pw:2 * pw]))
    print(torch.allclose(patches[gw], img[:, ph:2 * ph, 0:pw]))


@torch.no_grad()
def debug_show_image_and_patches(img, patches, patch_size, title="patch debug"):
    if img.dim() == 4:
        img = img[0]
    if patches.dim() == 5:
        patches = patches[0]

    img = img.detach().cpu()
    patches = patches.detach().cpu()

    C, H, W = img.shape
    ph, pw = patch_size
    gh = H // ph
    gw = W // pw
    expected = gh * gw

    if patches.shape[0] != expected:
        print(f"[warn] patchCount mismatch: got {patches.shape[0]}, expected {expected} (H//ph * W//pw)")
    if H % ph != 0 or W % pw != 0:
        print(f"[warn] image not divisible by patch size: H%ph={H%ph}, W%pw={W%pw}")

    # --- Reconstruct from patches (assumes row-major order: top-left -> right -> next row) ---
    recon = torch.zeros_like(img)
    idx = 0
    for iy in range(gh):
        for ix in range(gw):
            if idx >= patches.shape[0]:
                break
            recon[:, iy * ph:(iy + 1) * ph, ix * pw:(ix + 1) * pw] = patches[idx]
            idx += 1

    # Difference map (helps catch ordering / axis mistakes instantly)
    diff = (img - recon).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())

    # --- Plot original / recon / diff ---
    def to_hwc(x):
        x = x.permute(1, 2, 0)  # C,H,W -> H,W,C
        if x.shape[2] == 1:
            x = x.squeeze(2)
        return x

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"{title} | max_err={max_err:.6g}, mean_err={mean_err:.6g}")

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(to_hwc(img))

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.axis("off")
    plt.imshow(to_hwc(recon))

    plt.subplot(1, 3, 3)
    plt.title("Abs diff")
    plt.axis("off")
    # If RGB, show per-pixel max channel diff to make it readable
    diff_vis = diff.max(dim=0).values if diff.dim() == 3 and diff.shape[0] > 1 else diff.squeeze(0)
    plt.imshow(diff_vis)
    plt.show()

    # --- Plot all patches ---
    n = patches.shape[0]
    cols = gw
    rows = gh
    plt.figure(figsize=(cols * 2.0, rows * 2.0))
    plt.suptitle("All patches (row-major)")
    for i in range(min(n, rows * cols)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.axis("off")
        ax.imshow(to_hwc(patches[i]))
        ax.set_title(str(i), fontsize=8)
    plt.tight_layout()
    plt.show()