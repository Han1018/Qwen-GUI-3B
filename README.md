# Qwen-GUI-3B
Qwen-GUI-3B — A lightweight, resolution-aware GUI grounding model trained with only 24K samples on a single RTX 4090.

- [😊 Model Weights](https://huggingface.co/zonghanHZH/Qwen-GUI-3B)
- [😊 Demo](): Todo
- [😊 Training Data](): Todo
- [📑 Paper](): Todo


## 🔥 Updates
- [x] 2025/06/25: [`Qwen-GUI-3B`](https://huggingface.co/zonghanHZH/Qwen-GUI-3B) model weights are available on Hugging Face.


## Release Plans

- [x] [Model Weights]
- [ ] Online Demo (HF Spaces)
- [ ] Code
  - [ ] Experiments
    - [ ] ScreenSpot
    - [ ] ScreenSpotv2
    - [ ] ScreenSpotv2-pro
  - [ ] Inference Code
  - [ ] Training Code
- [ ] Paper
- [ ] Training-Data


## Main Results

### ScreenSpot

| Grounding Model          | Avg Score  | Mobile-Text | Mobile-Icon | Desktop-Text | Desktop-Icon | Web-Text | Web-Icon |
|--------------------------|--------|-------------|-------------|---------------|----------------|-----------|-----------|
| **General Models**       |        |             |             |               |                |           |           |
| Qwen2.5-VL-3B            | 55.5   | -           | -           | -             | -              | -         | -         |
| InternVL3-8B             | 79.5   | -           | -           | -             | -              | -         | -         |
| Claude3.5 Sonnet         | 83.0   | -           | -           | -             | -              | -         | -         |
| Gemini-2 Flash           | 84.0   | -           | -           | -             | -              | -         | -         |
| Qwen2.5-VL-7B            | 84.7   | -           | -           | -             | -              | -         | -         |
| **GUI-specific Models**  |        |             |             |               |                |           |           |
| CogAgent-18B             | 47.4   | 67.0        | 24.0        | 74.2          | 20.0           | 70.4      | 28.6      |
| SeeClick-9.6B            | 53.4   | 78.0        | 52.0        | 72.2          | 30.0           | 55.7      | 32.5      |
| OmniParser               | 73.0   | 93.9        | 57.0        | 91.3          | 63.6           | 81.3      | 51.0      |
| UGround-7B               | 73.3   | 82.8        | 60.3        | 82.5          | 63.6           | 80.4      | 70.4      |
| ShowUI-2B                | 75.0   | 91.6        | 69.0        | 81.8          | 59.0           | 83.0      | 65.5      |
| UI-TARS-2B               | 82.3   | 93.0        | 75.5        | 90.7          | 68.6           | 84.3      | 74.8      |
| OS-Atlas-7B              | 82.5   | 93.0        | 72.9        | 91.8          | 62.9           | 90.9      | 74.3      |
| Aguvis-7B                | 84.4   | 95.6        | 77.7        | 93.8          | 67.1           | 88.3      | 75.2      |
| **Qwen-GUI-3B**          | **84.9** | **96.3**    | **81.6**    | **93.8**      | **74.2**       | 89.5      | 74.2      |


### ScreenSpot-v2

| Grounding Model          | Avg Score  | Mobile-Text | Mobile-Icon | Desktop-Text | Desktop-Icon | Web-Text | Web-Icon |
|--------------------------|--------|-------------|-------------|---------------|----------------|-----------|-----------|
| **General Models**       |        |             |             |               |                |           |           |
| InternVL3-8B             | 81.4   | -           | -           | -             | -              | -         | -         |
| **GUI-specific Models**  |        |             |             |               |                |           |           |
| SeeClick-9.6B            | 55.1   | 78.4        | 50.7        | 70.1          | 29.3           | 55.2      | 32.5      |
| UGround-7B               | 76.3   | 84.5        | 61.6        | 85.1          | 61.4           | 84.6      | 71.9      |
| ShowUI-2B                | 77.3   | 92.1        | 75.4        | 78.9          | 59.3           | 84.2      | 61.1      |
| OS-Atlas-7B              | 84.1   | 95.1        | 75.8        | 90.7          | 63.5           | 90.6      | 77.3      |
| UI-TARS-2B               | 84.7   | 95.2        | 79.1        | 90.7          | 68.6           | 87.2      | 78.3      |
| **Qwen-GUI-3B**        | **86.4** | **97.9**    | **84.8**    | **93.8**      | **75.0**       | **91.0**  | 75.8      |


### ScreenSpot-Pro
Todo

## Hardware
- GPU: 1 × RTX 4090
- Time: <= 48 hrs
