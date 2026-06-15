# Benchmark Summary

## Environment
- Python 3.12.2, Windows-10-10.0.19045-SP0, CPU threads: 16
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD

## Presentation Numbers
- Warm full pipeline mean: 604.9 ms
- Warm full pipeline median: 437.7 ms
- Warm full pipeline p95: 1435.0 ms
- Fastest camera: zaki_validi_aksakova (167.6 ms, 15 slots)
- Slowest camera: salavat_square (1466.9 ms, 75 slots)
- Vehicle detector mean per call: 15.14 ms
- Slot classifier mean per call: 6.84 ms
- Cold start reference: 389.0 ms on mendeleeva_130_cam_5 (9 slots)
- mAP50: 0.9946
- mAP50-95: 0.9821
- Precision: 0.9984
- Recall: 0.9985

## Per Camera
- salavat_square: mean=1466.9 ms, p95=1551.3 ms, slots=75
- bakalinskaya_27_vk6: mean=1234.5 ms, p95=1256.3 ms, slots=57
- bakalinskaya_27_vk7: mean=951.5 ms, p95=962.4 ms, slots=35
- mendeleeva_130_cam_9: mean=662.8 ms, p95=670.0 ms, slots=28
- ufa_arena_1: mean=443.6 ms, p95=450.5 ms, slots=26
- sports_palace_square: mean=440.0 ms, p95=442.7 ms, slots=22
- park_pobedy: mean=346.8 ms, p95=357.7 ms, slots=16
- lesotehnikuma_luganskaya: mean=338.0 ms, p95=342.1 ms, slots=30
- mushnikova_balandina: mean=327.0 ms, p95=330.5 ms, slots=21
- mendeleeva_130_cam_5: mean=275.5 ms, p95=279.9 ms, slots=9
- zaki_validi_aksakova: mean=167.6 ms, p95=170.9 ms, slots=15
