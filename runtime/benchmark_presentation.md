# Benchmark Summary

## Environment
- Python 3.12.3, Linux-6.17.0-35-generic-x86_64-with-glibc2.39, CPU threads: 16
- Processor: x86_64

## Presentation Numbers
- Warm full pipeline mean: 2414.0 ms
- Warm full pipeline median: 1496.6 ms
- Warm full pipeline p95: 5781.8 ms
- Fastest camera: zaki_validi_aksakova (239.7 ms, 15 slots)
- Slowest camera: bakalinskaya_27_vk6 (5898.1 ms, 57 slots)
- Vehicle detector mean per call: 92.08 ms
- Slot classifier mean per call: 7.31 ms
- Cold start reference: 1193.4 ms on mendeleeva_130_cam_5 (9 slots)
- mAP50: 0.9946
- mAP50-95: 0.9821
- Precision: 0.9984
- Recall: 0.9985

## Per Camera
- bakalinskaya_27_vk6: mean=5898.1 ms, p95=6064.8 ms, slots=57
- salavat_square: mean=5012.5 ms, p95=5120.1 ms, slots=75
- bakalinskaya_27_vk7: mean=4752.9 ms, p95=4975.0 ms, slots=35
- mendeleeva_130_cam_9: mean=3326.1 ms, p95=3339.4 ms, slots=28
- ufa_arena_1: mean=2575.4 ms, p95=2695.0 ms, slots=26
- park_pobedy: mean=1493.0 ms, p95=1509.9 ms, slots=16
- sports_palace_square: mean=1210.4 ms, p95=1225.4 ms, slots=22
- mendeleeva_130_cam_5: mean=1055.4 ms, p95=1071.1 ms, slots=9
- mushnikova_balandina: mean=573.8 ms, p95=575.8 ms, slots=21
- lesotehnikuma_luganskaya: mean=416.2 ms, p95=421.9 ms, slots=30
- zaki_validi_aksakova: mean=239.7 ms, p95=259.3 ms, slots=15
