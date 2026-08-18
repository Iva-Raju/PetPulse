# PetPulse: Real-Time Stress & Injury Detection for Pets

An AI-driven system that monitors pet wellbeing in real time by analyzing both audio and video — detecting stress through vocalizations and identifying wounds through visual inspection.

## Problem

Early signs of pet stress or injury are easily missed by owners, and manual monitoring is time-consuming and subjective. PetPulse automates this using multimodal AI.

## Features

- **Audio Intelligence**: Captures live audio (16kHz), extracts MFCC/spectral features via Librosa, and classifies stress using a CNN-LSTM hybrid model
- **Vision Intelligence**: Detects and classifies wound severity (mild/severe) from video/images using YOLOv8
- **Fusion Engine**: Combines audio and vision outputs into an overall Pet Health Score with anomaly detection
- **Event Logging**: Tracks detection history with timestamps and frequency monitoring
- **Recommendations**: Generates actionable care/veterinary guidance based on results

## Tech Stack

- **Backend**: Python, SQLite, FastAPI
- **ML/AI**: TensorFlow/Keras (CNN-LSTM), YOLOv8, Librosa, NumPy
- **Frontend**: HTML, CSS, JavaScript

## Datasets

- **Audio**: animal_mood_dataset (GitHub) — used for training the stress classification model
- **Images**: Roboflow Injured Dog Detector — used for training the injury detection model

## Performance

| Model | Key Metric | Score |
|---|---|---|
| Stress Detection (CNN-LSTM) | Accuracy | 89% |
| Wound Detection (YOLOv8) | mAP50 | 0.76 |

## System Requirements

**Software**: Windows 10+, Python, Visual Studio Code
**Hardware**: Intel Core i3 or higher, 8GB+ RAM, 256GB SSD, webcam, microphone
