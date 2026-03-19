# BCI-Agent-Bridge

A brain-computer interface pipeline that translates simulated EEG neural signals into LLM agent commands.

## What it does

```
raw EEG (channels × time)
    ↓  band-pass filter  (alpha / beta / theta)
    ↓  FFT power features
    ↓  differential-privacy noise
    ↓  linear intent classifier
    ↓  agent command mapper
AgentCommand  →  action + instruction for LLM agent
```

### Intents → Actions

| Mental State | EEG Signature          | Intent   | Agent Action |
|--------------|------------------------|----------|--------------|
| Focused work | elevated beta (13–30 Hz) | FOCUS  | `search`     |
| Relaxed/idle | elevated alpha (8–13 Hz) | RELAX  | `idle`       |
| Left imagery | left-lateralised beta  | LEFT     | `previous`   |
| Right imagery| right-lateralised beta  | RIGHT   | `next`       |
| Decision     | elevated theta (4–8 Hz) | SELECT  | `execute`    |

## Quick start

```bash
pip install -e .
```

```python
from bci_agent_bridge import BCIAgentBridge

bridge = BCIAgentBridge()
results = bridge.run_demo(duration_secs=10, n_frames=10)
```

Example output:

```
──────────────────────────────────────────────────────────────────────
  BCIAgentBridge Demo  |  10 frames × 1.0s  |  DP ε=1.0
──────────────────────────────────────────────────────────────────────
Frame 00 | GT=FOCUS  | Pred=FOCUS  ✓ | Conf=74.12% | Action='search'
Frame 01 | GT=RELAX  | Pred=RELAX  ✓ | Conf=81.33% | Action='idle'
Frame 02 | GT=LEFT   | Pred=LEFT   ✓ | Conf=66.50% | Action='previous'
...
──────────────────────────────────────────────────────────────────────
  Accuracy: 8/10 (80%)
  Privacy:  Gaussian DP active
──────────────────────────────────────────────────────────────────────
```

## Architecture

### `EEGSignalProcessor`

Takes a `(n_channels, n_timepoints)` numpy array and returns `EEGFeatures`:

- Butterworth band-pass filters for theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz)
- FFT-based mean power per band per channel
- Flat feature vector of shape `(3 × n_channels,)`

```python
from bci_agent_bridge import EEGSignalProcessor
import numpy as np

proc = EEGSignalProcessor(sampling_rate=256)
raw = np.random.randn(8, 1024)   # 8 channels, 4 seconds
features = proc.process(raw)
print(features.band_powers)  # {'theta': ..., 'alpha': ..., 'beta': ...}
```

### `IntentClassifier`

Trains a linear softmax classifier on synthetic labeled EEG features at construction time. Each intent class has a characteristic spectral signature (e.g. FOCUS = high beta, RELAX = high alpha).

Differential privacy (Gaussian mechanism, default ε=1.0, δ=1e-5) is injected into the feature vector before classification to prevent neural fingerprinting.

```python
from bci_agent_bridge import IntentClassifier, Intent

clf = IntentClassifier(n_channels=8, epsilon=1.0)
result = clf.classify(features, apply_privacy=True)
print(result.intent, result.confidence)   # Intent.FOCUS  0.74
```

### `AgentCommandMapper`

Maps `ClassificationResult → AgentCommand`. Includes a confidence threshold — low-confidence signals fall back to `idle` rather than emitting noisy commands.

```python
from bci_agent_bridge import AgentCommandMapper

mapper = AgentCommandMapper(confidence_threshold=0.35)
command = mapper.map(result)
print(command.action)       # 'search'
print(command.instruction)  # 'Search for information on the current topic.'
```

### `BCIAgentBridge`

Orchestrates the full pipeline. Also generates synthetic EEG with realistic oscillations per intent for demo/testing.

```python
from bci_agent_bridge import BCIAgentBridge
import numpy as np

bridge = BCIAgentBridge(n_channels=8, sampling_rate=256, epsilon=1.0)

# Process a real EEG frame
raw_eeg = np.random.randn(8, 256)   # 1 second
result = bridge.process_frame(raw_eeg)
print(result.command)
```

## Privacy

Features are privatised using the **Gaussian mechanism** before classification:

- Noise scale: `σ = √(2 ln(1.25/δ)) × Δf / ε`
- Default: ε=1.0, δ=1e-5
- Set `epsilon=0` to disable DP

Lower ε = stronger privacy, higher noise, lower accuracy.

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

## Dependencies

- `numpy`
- `scipy`

No PyTorch, no ML frameworks, no cloud dependencies.

## License

MIT
