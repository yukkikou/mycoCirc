#!/bin/bash
# Save from-scratch results, then restore pretrained checkpoints
echo "=========================================="
echo "  Saving from-scratch results"
echo "=========================================="

FINETUNE_DIR="/media/share/workdir/1610305236/Panfungi/4_model/checkpoints/finetune"

for GROUP in Cryptococcus Filamentous; do
  CKPT="$FINETUNE_DIR/$GROUP/final.pt"
  SAVE="$FINETUNE_DIR/$GROUP/final_fromscratch.pt"
  RESTORE="$FINETUNE_DIR/$GROUP/final_pretrained.pt"

  if [ -f "$CKPT" ]; then
    echo "  $GROUP: saving from-scratch → final_fromscratch.pt"
    cp "$CKPT" "$SAVE"
  fi

  if [ -f "$RESTORE" ]; then
    echo "  $GROUP: restoring pretrained → final.pt"
    cp "$RESTORE" "$CKPT"
    echo "  $GROUP: done"
  else
    echo "  $GROUP: no pretrained backup found!"
  fi
done

echo "=========================================="
echo "  Complete"
echo "=========================================="
