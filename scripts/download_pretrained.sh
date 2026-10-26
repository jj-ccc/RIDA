#!/usr/bin/env bash
# Foundation-model weights used by the DINOv3 and MedSAM settings.
#   pretrained/dinov3_vitl16_pretrain_lvd1689m.pth  DINOv3 ViT-L/16 (LVD-1689M), released by Meta AI after
#                                                   accepting the DINOv3 license:
#                                                   https://github.com/facebookresearch/dinov3
#   pretrained/medsam_vit_b.pth                     MedSAM ViT-B checkpoint:
#                                                   https://github.com/bowang-lab/MedSAM
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pretrained
if [ -n "${DINOV3_URL:-}" ]; then
  wget -c "${DINOV3_URL}" -O pretrained/dinov3_vitl16_pretrain_lvd1689m.pth
fi
if [ -n "${MEDSAM_URL:-}" ]; then
  wget -c "${MEDSAM_URL}" -O pretrained/medsam_vit_b.pth
fi
for f in pretrained/dinov3_vitl16_pretrain_lvd1689m.pth pretrained/medsam_vit_b.pth; do
  if [ -f "$f" ]; then echo "[ok] $f"; else echo "[missing] $f (set DINOV3_URL / MEDSAM_URL or place the file manually)"; fi
done
