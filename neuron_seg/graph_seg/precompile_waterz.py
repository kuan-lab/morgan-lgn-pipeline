"""Compile the waterz scoring module the edge step needs (aff50_his256, queue 256) into this job's own ~/.cython cache.
Tim's waterz copy assumes it is already compiled unless told to rebuild, and the workers do not ask for a rebuild."""
import numpy as np
from waterz import agglomerate
from magneton.instance_segmentation.stages.edges_stage import getScoreFunc
aff = np.random.rand(3, 4, 16, 16).astype(np.float32); frag = (np.arange(4 * 16 * 16, dtype=np.uint64).reshape(4, 16, 16) // 8) + 1
for _ in agglomerate(aff, [0.5], fragments=frag, aff_threshold_low=0.00001, aff_threshold_high=0.99999, scoring_function=getScoreFunc('aff50_his256'), discretize_queue=256, force_rebuild=True): pass
print('waterz scoring module compiled', flush=True)
