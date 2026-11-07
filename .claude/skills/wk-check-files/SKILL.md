---
name: wk-check-files
description: Write webKnossos NML files (landmarks, synapses, merge sites) that pass wk's strict parser, plus a matching Excel sheet with a verdict column for checking by eye. Use whenever Nghia asks for an NML, landmarks, a sheet/excel to check, or coordinates to look at.
---

# NML + check sheet for Nghia

Template generator: `/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/make_check_v3_v4.py`. It makes top
sites plus a random sample per group with a control group, and the NML tree names match the sheet IDs. Copy it for new
checks.

## NML rules (wk rejects the file otherwise)

- No `<?xml ...?>` line. Root `<things>`, then `<meta name="writer" content="..." />`, then `<parameters>` with
  `<experiment name="Morgan" .../>`, `<scale x="4" y="4" z="30" unit="nanometer" />`, `<offset>`, `<editPosition>`,
  `<editRotation>` and `<zoomLevel>`.
- Every `<thing>` needs `id`, `groupId`, `name`, and `color.r/g/b/a` (0–1). Put the differentiator in `name`: it's how
  Nghia navigates the tree list.
- Node IDs are unique across the file. Self-closing tags have a space before `/>`.
- Always end with `<branchpoints></branchpoints>`, `<comments></comments>`, and `<groups>`, with one `<group>` per
  `groupId` used.
- Coordinates are wk (4 × 4 × 30) unless Nghia asks for "bbox coords" (local 8 × 8 × 30, `scale 8 8 30`).
- Verify with `xml.etree` (tree count, unique node IDs, coordinate ranges) before handing it over.

## Sheet

- Columns: id, group, wk as a `"x, y, z"` string, distance to the skeleton, the layer to view, segment IDs as strings,
  `verdict` (a dropdown) and `notes`. Add a "how to" tab.
- openpyxl isn't in our envs: `python3 -m pip install --target <scratchpad>/xl openpyxl`, then run with that folder on
  `PYTHONPATH`.

## Hand-over

Give the **full absolute path of both files** (Nghia can't find files otherwise), what each group is for, and which
segmentation layer to load. Coordinates in chat are always `x, y, z`, ready to paste.
