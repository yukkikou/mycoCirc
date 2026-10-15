#!/bin/bash
# Wait for rescore job then clean stale P4 files (from cancelled 27587) and verify
JID=$1
while squeue -h -j "$JID" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$JID" 2>/dev/null)" ]; do sleep 60; done
# P4 npz from 16:48/16:49 are stale (pre-fix); remove if older than C4's
for f in /media/share/workdir/1610305236/Panfungi/4_model/revision/results/rescore/heldout_P4_*.npz; do
  if [ "$(stat -c %Y "$f")" -lt "$(stat -c %Y /media/share/workdir/1610305236/Panfungi/4_model/revision/results/rescore/heldout_C4_Candida_final.pt.npz 2>/dev/null || echo 0)" ]; then
    rm -f "$f"; echo "removed stale $f"
  fi
done
echo "job $JID finished"
