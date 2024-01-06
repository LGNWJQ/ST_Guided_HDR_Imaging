# ST_Guided_HDR_Imaging


## dataset
- batch: ldr1, ldr2, hdr, hdr_tmap
- dtype: floar32
- range: [0, 1]
- Train: random crop
- Test/Val: center crop

## model
- 3stage
  1. encoder: encode 2 ldr image as feature
  2. decoder: decode feature as hdr
  3. tmap-decoder: tone-map hdr as ldr

- 2branch
  1. encoder: encode 2 ldr image as feature
  2. decoder1: decode feature as hdr
  3. decoder2: decode feature as ldr(tone-map from hdr)

## loss
- hdr:
    1. $\mu$ law -> L1
    1. $\mu$ law -> LPIPS
- ldr:
    1. L1
    2. LPIPS
