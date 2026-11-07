"""Which sections of an EM tile have no image content: black, or structureless (saturated white, grey noise, washed out).

Black frames were the only kind handled until 2026-10-08; wk z 2355-2357 and 7617 are white / noise frames and the
models hallucinate on them exactly as on black ones.
"""
import numpy as np

BLANK_ZERO_FRAC, BLANK_MEAN = 0.5, 40
# structureless frame = a short dropout to near-zero contrast between normal sections. Checked by eye 2026-10-08:
# a looser rule (below 0.6 x the tile median) also caught real low-contrast sections (wk z 940, 4373) and vessel lumen.
DROP_REL, DROP_ABS, CONTENT_REL, MIN_REF, MAX_RUN = 0.25, 10.0, 0.6, 25.0, 3
FORCE_EMPTY_LOCAL_Z = (2221, 2222, 2223, 7483)     # wk 2355, 2356, 2357, 7617: empty over (nearly) the whole section


def contrast64(core):
    """std of each section after 8x8 averaging (8 nm -> 64 nm): real EM keeps ~35-65, a noise frame 3-13. core = (x, y, z) uint8"""
    X, Y, Z = core.shape
    return core[:X // 8 * 8, :Y // 8 * 8].reshape(X // 8, 8, Y // 8, 8, Z).mean((1, 3), dtype=np.float32).std((0, 1))


def dropouts(h, black):
    """sections (not black) whose contrast drops to ~0 for <= MAX_RUN consecutive sections, with normal sections on both sides"""
    low = np.zeros(len(h), bool); ok = np.nonzero(~black)[0]
    if len(ok) < 8: return low
    ref = np.median(h[ok])
    if ref < MIN_REF: return low                       # uniform region (vessel lumen ...): contrast says nothing here
    cand = (h[ok] < DROP_REL * ref) & (h[ok] < DROP_ABS); content = h[ok] >= CONTENT_REL * ref
    k = 0
    while k < len(ok):
        if not cand[k]: k += 1; continue
        a = k
        while k < len(ok) and cand[k]: k += 1
        below = content[a - 1] if a > 0 else None; above = content[k] if k < len(ok) else None
        sides = [v for v in (below, above) if v is not None]
        if k - a <= MAX_RUN and sides and all(sides): low[ok[a:k]] = True
    return low


def empty_sections(core, zs=None):
    """(empty, black, contrast) per section. zs = local z of each section (for the forced list)."""
    black = ((core == 0).mean(axis=(0, 1)) > BLANK_ZERO_FRAC) | (core.mean(axis=(0, 1)) < BLANK_MEAN)
    h = contrast64(core); low = dropouts(h, black)
    if zs is not None:
        low |= ~black & np.isin(zs, FORCE_EMPTY_LOCAL_Z)
    return black | low, black, h
