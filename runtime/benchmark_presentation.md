# Benchmark Summary

## Environment
- Python 3.12.3, Linux-6.17.0-35-generic-x86_64-with-glibc2.39, CPU threads: 16
- Processor: x86_64

## Presentation Numbers
- Warm full pipeline mean: 2503.7 ms
- Warm full pipeline median: 1581.0 ms
- Warm full pipeline p95: 6010.1 ms
- Fastest camera: zaki_validi_aksakova (255.8 ms, 15 slots)
- Slowest camera: bakalinskaya_27_vk6 (6032.8 ms, 57 slots)
- Vehicle detector mean per call: 94.47 ms
- Slot classifier mean per call: 8.51 ms
- Cold start reference: 1395.0 ms on mendeleeva_130_cam_5 (9 slots)
- mAP50: 0.9946
- mAP50-95: 0.9821
- Precision: 0.9984
- Recall: 0.9985

## Per Camera
- bakalinskaya_27_vk6: mean=6032.8 ms, p95=6083.7 ms, slots=57
- salavat_square: mean=5527.1 ms, p95=5550.8 ms, slots=75
- bakalinskaya_27_vk7: mean=5180.4 ms, p95=5215.7 ms, slots=35
- mendeleeva_130_cam_9: mean=3330.9 ms, p95=3347.8 ms, slots=28
- ufa_arena_1: mean=2084.7 ms, p95=2096.7 ms, slots=26
- park_pobedy: mean=1584.6 ms, p95=1607.5 ms, slots=16
- sports_palace_square: mean=1301.6 ms, p95=1306.3 ms, slots=22
- mendeleeva_130_cam_5: mean=1176.9 ms, p95=1213.8 ms, slots=9
- mushnikova_balandina: mean=617.7 ms, p95=620.8 ms, slots=21
- lesotehnikuma_luganskaya: mean=448.3 ms, p95=451.3 ms, slots=30
- zaki_validi_aksakova: mean=255.8 ms, p95=266.9 ms, slots=15
