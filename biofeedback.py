import serial
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from collections import deque

# ---- Config ----
PORT = '/dev/cu.usbserial-130'   # update after `ls /dev/cu.*`
BAUD = 9600                       # match Serial.begin(...) in your Arduino sketch
WINDOW_SIZE = 50

CLASS_WEIGHTS = {'rest': 0.0, 'light_flex': 0.5, 'hard_flex': 1.0}
CLASS_LABELS = {'rest': 'Rest', 'light_flex': 'Light flex', 'hard_flex': 'Hard flex'}

ZONES = {
    'rest':       {'fill': '#639922', 'text': '#3B6D11', 'badge_bg': '#EAF3DE'},
    'light_flex': {'fill': '#BA7517', 'text': '#854F0B', 'badge_bg': '#FAEEDA'},
    'hard_flex':  {'fill': '#A32D2D', 'text': '#791F1F', 'badge_bg': '#FCEBEB'},
}

# ---- Load trained model ----
model = joblib.load('emg_model.pkl')
weight_vector = np.array([CLASS_WEIGHTS[c] for c in model.classes_])


# ---- Feature extraction (matches process_data.py: MAV, RMS, ZC, WL, STD) ----
def extract_features(window):
    window = np.array(window)
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    mean_val = np.mean(window)
    zc = np.sum(np.diff(np.sign(window - mean_val)) != 0)
    wl = np.sum(np.abs(np.diff(window)))
    std = np.std(window)
    return [mav, rms, zc, wl, std]


def score_and_class(window):
    features = np.array(extract_features(window)).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    score = float(np.dot(probs, weight_vector))
    predicted_class = model.classes_[np.argmax(probs)]
    return score, predicted_class


# ---- Figure setup ----
plt.rcParams['toolbar'] = 'None'   # hide the magnifier/pan/save icon bar
fig, ax = plt.subplots(figsize=(6.4, 3.4))
try:
    fig.canvas.manager.set_window_title('EMG activation')
except Exception:
    pass
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)

TRACK_Y0, TRACK_Y1 = 0.30, 0.58   # vertical extent of the track/bands
TRACK_H = TRACK_Y1 - TRACK_Y0
BAR_PAD = 0.035                   # inset of the pill bar within the track
bar_h = TRACK_H - 2 * BAR_PAD

# Rounded-pill clip shape for the whole track (invisible itself — just a mask)
track_clip = FancyBboxPatch((0.0, TRACK_Y0), 1.0, TRACK_H,
                             boxstyle=f"round,pad=0,rounding_size={TRACK_H/2:.3f}",
                             transform=ax.transData, facecolor='none', edgecolor='none')
ax.add_patch(track_clip)

# Zone background bands, clipped to the rounded track so corners match the bar
for x0, x1, color in [(0.00, 0.34, '#EAF3DE'), (0.34, 0.67, '#FAEEDA'), (0.67, 1.00, '#FCEBEB')]:
    zone_rect = Rectangle((x0, TRACK_Y0), x1 - x0, TRACK_H, facecolor=color, edgecolor='none', zorder=0)
    ax.add_patch(zone_rect)
    zone_rect.set_clip_path(track_clip)

# Zone labels, below the track
ax.text(0.17, TRACK_Y0 - 0.06, 'Rest', ha='center', va='top', fontsize=11, color='#888780')
ax.text(0.50, TRACK_Y0 - 0.06, 'Light flex', ha='center', va='top', fontsize=11, color='#888780')
ax.text(0.835, TRACK_Y0 - 0.06, 'Hard flex', ha='center', va='top', fontsize=11, color='#888780')

# Big percentage readout
pct_text = ax.text(0.5, 0.88, '0%', ha='center', va='center',
                    fontsize=42, fontweight='medium', color=ZONES['rest']['text'])

# State badge (colored pill behind the state label)
badge = FancyBboxPatch((0.37, 0.655), 0.26, 0.11,
                        boxstyle="round,pad=0,rounding_size=0.055",
                        linewidth=0, facecolor=ZONES['rest']['badge_bg'], zorder=3)
ax.add_patch(badge)
badge_text = ax.text(0.5, 0.71, 'Rest', ha='center', va='center',
                      fontsize=14, fontweight='medium', color=ZONES['rest']['text'], zorder=4)

# Rounded activation bar (recreated on each update — FancyBboxPatch has no simple set_width)
bar = FancyBboxPatch((0.0, TRACK_Y0 + BAR_PAD), 0.02, bar_h,
                      boxstyle=f"round,pad=0,rounding_size={bar_h/2:.3f}",
                      linewidth=0, facecolor=ZONES['rest']['fill'], zorder=2)
ax.add_patch(bar)

plt.show(block=False)
plt.pause(0.01)

# ---- Serial + main loop ----
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # let the Arduino reset after opening the port
buffer = deque(maxlen=WINDOW_SIZE)

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        try:
            value = float(line)
        except ValueError:
            continue

        buffer.append(value)

        if len(buffer) == WINDOW_SIZE:
            score, predicted_class = score_and_class(list(buffer))
            colors = ZONES[predicted_class]
            width = max(score, 0.02)  # keep a visible sliver even at 0%

            bar.remove()
            bar = FancyBboxPatch((0.0, TRACK_Y0 + BAR_PAD), width, bar_h,
                                  boxstyle=f"round,pad=0,rounding_size={bar_h/2:.3f}",
                                  linewidth=0, facecolor=colors['fill'], zorder=2)
            ax.add_patch(bar)

            pct_text.set_text(f'{int(round(score * 100))}%')
            pct_text.set_color(colors['text'])

            badge.set_facecolor(colors['badge_bg'])
            badge_text.set_text(CLASS_LABELS[predicted_class])
            badge_text.set_color(colors['text'])

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print('Stopped.')
finally:
    ser.close()