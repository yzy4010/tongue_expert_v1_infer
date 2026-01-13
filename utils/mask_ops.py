from __future__ import annotations
import numpy as np

# ===== Fixed class ids (MUST match training) =====
CLS_BG = 0
CLS_TONGUE = 1
CLS_TAI = 2
CLS_ZHI = 3
CLS_FISSURE = 4
CLS_TOOTH_MK = 5


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return uint8 {0,1} mask."""
    if mask is None:
        raise ValueError("mask is None")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def apply_tongue_constraint(pred_cls: np.ndarray, tongue_mask: np.ndarray) -> np.ndarray:
    """
    Force predictions outside tongue_mask to BG.
    pred_cls: HxW int (0..5)
    tongue_mask: HxW {0,1}
    """
    tongue_mask = ensure_binary_mask(tongue_mask)
    out = pred_cls.copy()
    out[tongue_mask == 0] = CLS_BG
    return out


def fuse_by_priority(
    pred_cls: np.ndarray,
    tongue_mask: np.ndarray,
    fill_zhi_from_tongue: bool = True,
) -> dict[str, np.ndarray]:
    """
    Create final binary roi masks following fixed priority:
      tooth_mk > fissure > tai > zhi > tongue > bg

    Returns dict of {roi_name: HxW uint8(0/1)}
    """
    tongue_mask = ensure_binary_mask(tongue_mask)
    pred_cls = apply_tongue_constraint(pred_cls, tongue_mask)

    tooth = (pred_cls == CLS_TOOTH_MK).astype(np.uint8)
    fissure = ((pred_cls == CLS_FISSURE) & (tooth == 0)).astype(np.uint8)
    tai = ((pred_cls == CLS_TAI) & (tooth == 0) & (fissure == 0)).astype(np.uint8)
    zhi = ((pred_cls == CLS_ZHI) & (tooth == 0) & (fissure == 0) & (tai == 0)).astype(np.uint8)

    # Optional but recommended: if zhi is empty/sparse, fill as tongue minus (tai/fissure/tooth).
    if fill_zhi_from_tongue:
        zhi_fill = (tongue_mask == 1) & (tai == 0) & (fissure == 0) & (tooth == 0)
        zhi = (zhi | zhi_fill.astype(np.uint8)).astype(np.uint8)

    return {
        "tongue": tongue_mask.astype(np.uint8),
        "tai": tai,
        "zhi": zhi,
        "fissure": fissure,
        "tooth_mk": tooth,
    }
