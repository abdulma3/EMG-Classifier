# EMG-Classifier
# EMG muscle activation classifier

A real-time system that reads electrical activity from a flexing muscle via a surface EMG sensor and classifies exertion level (rest, light flex, hard flex) using a trained machine learning model — with a live biofeedback gauge built on top of it.

## What it does

A MyoWare 2.0 Muscle Sensor picks up electrical activity from the bicep and streams it to an Arduino Nano, which forwards raw readings over serial to a Python pipeline. The pipeline extracts signal features from short windows of that data and feeds them into a Random Forest classifier trained to recognize three states: **rest**, **light flex**, and **hard flex**. A separate real-time gauge turns that classification into continuous visual feedback — similar in principle to EMG biofeedback tools used in physical therapy for muscle re-education.

## Hardware

- MyoWare 2.0 Muscle Sensor (SparkFun DEV-27924)
- Arduino Nano
- Disposable surface EMG electrode pads, placed on the bicep belly
- Breadboard + jumper wires

Wiring: Nano `5V` → sensor `VIN`, Nano `GND` → sensor `GND`, Nano `A0` → sensor `ENV` (the smoothed/rectified envelope output).

## Pipeline

| File | Purpose |
|---|---|
| `collect_data.py` | Records labeled EMG sessions (rest / light_flex / hard_flex) to `emg_data.csv` |
| `process_data.py` | Extracts 5 features per 50-sample window — MAV, RMS, ZC, WL, STD — into `emg_features.csv` |
| `train_classifier.py` | Trains a Random Forest classifier on the extracted features, reports accuracy and cross-validation, saves `emg_model.pkl` |
| `live_predict.py` | Real-time single-label prediction from live sensor input |
| `biofeedback.py` | Real-time continuous activation gauge (0-100%) using the trained model's class probabilities |
| `check_rate.py` | Diagnostic script confirming real hardware sampling rate (~154 samples/sec) |

## Results

The model's accuracy improved through several rounds of debugging and iteration, not a single training run:

| Stage | Result |
|---|---|
| First trial (forearm placement, buffer bug present) | 38% accuracy |
| After bicep placement + buffer fix + feature engineering | 56-62% (single split) |
| After 10 reps of data collection | 66% (single split), 52.5% ±8.9% (cross-validated) |
| Final | 66% (single split), **54.3% ±6.5% (cross-validated)** |

The cross-validated number is the one to trust — a single train/test split can look better or worse by luck, and doesn't tell you how the model performs on data it hasn't seen in a specific arrangement.

**A significant early result was not a training improvement, but a data-integrity fix.** The first version of the data collection script had a buffer bug: because the Arduino streams data continuously, leftover signal from one label was bleeding into the next during rest periods between reps, silently mislabeling training data. Fixing this — not further model tuning — was what took accuracy from 38% to the 50s.

Feature importance was roughly even across MAV, RMS, WL, and STD (~0.19-0.28 each); zero-crossing (ZC) contributed least (~0.10), which is expected since ZC is better suited to raw oscillating signal than the already-smoothed ENV output used here.

## Biofeedback trainer

`biofeedback.py` reuses the trained classifier but reframes its output: instead of a discrete label, it computes a continuous 0-100% activation score (a probability-weighted blend across the three classes) and displays it on a live, color-coded gauge.

This mirrors the real-time feedback loop used in clinical EMG biofeedback for muscle re-education — for example, post-surgical quadriceps activation or post-stroke motor recovery, where a patient uses a visual signal of their own muscle activity to relearn graduated control. This project is a personal proof-of-concept demonstrating that sensing-and-feedback mechanism, not a validated or tested rehabilitation device.

## Setup

1. Wire the hardware as described above and confirm the sensor is reading cleanly (see `check_rate.py`).
2. Install dependencies: `pip3 install pyserial numpy scipy matplotlib pandas scikit-learn joblib`
3. Update the `PORT` variable in each script to match your Arduino's current serial port (`ls /dev/cu.*` on Mac, or check the Arduino IDE's Tools → Port).
4. Collect data: `python3 collect_data.py`
5. Extract features: `python3 process_data.py`
6. Train the model: `python3 train_classifier.py`
7. Run live prediction or the biofeedback gauge: `python3 live_predict.py` or `python3 biofeedback.py`

## Future work

- **Muscle fatigue detection** — EMG frequency content shifts in a known way as a muscle fatigues during sustained contraction. This would require reading the sensor's raw (unfiltered) output and adding frequency-domain analysis (FFT), rather than the time-domain features used here.
- **True gesture recognition** — distinguishing specific hand/arm movements, not just intensity levels, would need a second sensor on a different muscle group and a redesigned data collection protocol.