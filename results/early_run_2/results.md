# Experimental results

All values below are read from saved experiment outputs.

| Method | Test IoU | Test F1 | Precision | Recall | Threshold | Parameters | Train time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB difference | 0.0700 | 0.1308 | 0.1040 | 0.1760 | 0.40 | — | — |
| Early fusion U-Net | 0.5951 | 0.7461 | 0.7784 | 0.7164 | 0.30 | 7,763,905 | 206.1 |

Thresholds were chosen on validation data only. Test metrics are global pixel counts (micro average).
