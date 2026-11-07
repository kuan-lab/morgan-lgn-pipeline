"""Submit boxes: GPU job (fetch + inference), then CPU job (seg + pairing) chained with afterok.

  python submit.py                 # every box in boxes.csv
  python submit.py d050_a d100_b   # just these
  python submit.py --cpu-only X    # rerun step 3 only (aff/syn_prob already exist)
"""
import subprocess
import sys

import config as C


def sbatch(script, box, dep=None):
    cmd = ['sbatch', '--parsable', f'--export=ALL,BOX={box}', f'--job-name={script[4:7]}_{box}']
    if dep:
        cmd.append(f'--dependency=afterok:{dep}')
    return subprocess.check_output(cmd + [script], cwd=C.ROOT, text=True).strip()


argv = sys.argv[1:]
cpu_only = '--cpu-only' in argv
names = [a for a in argv if not a.startswith('--')] or list(C.load_boxes())
for name in names:
    gpu = None if cpu_only else sbatch('job_gpu.sh', name)
    cpu = sbatch('job_cpu.sh', name, dep=gpu)
    print(f'{name}: gpu={gpu or "-"} cpu={cpu}')
