# Vectorized Dueling DQN Pong with GPU-Resident Replay Buffer

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg)](https://pytorch.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.6%20%2B-green.svg)](https://www.pygame.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieses Repository enthält eine hochgradig optimierte Implementierung eines Dueling Deep Q-Networks (DQN), das darauf trainiert wird, das klassische Spiel Pong zu meistern. Die Trainingspipeline nutzt eine vollständig vektorisierte NumPy-Umgebung sowie einen GPU-residenten Replay-Buffer, um die Latenzzeiten beim Daten-Transfer zu minimieren und moderne Multi-Core-CPUs sowie GPUs maximal auszulasten.

---

## Hauptfeatures

* **Vektorisierte Umgebung (`VecPongEnv`)**: Erlaubt die gleichzeitige Ausführung von hunderten oder tausenden Pong-Instanzen in einem einzigen Python-Prozess mittels hocheffizienter NumPy-Vektorisierung.
* **GPU-Residenter Replay-Buffer**: Eliminiert den PCIe-Flaschenhals. Alle Zustandsübergänge werden direkt auf der GPU (CUDA/ROCm) gehalten, indiziert und gesampelt, wodurch der Datentransfer zwischen Host und Device drastisch reduziert wird.
* **Dueling DQN Architektur**: Trennt die Schätzung des Zustandswertes (State Value) und des Vorteils einzelner Aktionen (Advantage Stream) für stabileres Lernen.
* **Automatische Breiten-Erkennung**: Beim Laden von Checkpoints werden die Dimensionen der verborgenen Schichten automatisch aus den Gewichtungsmatrizen rekonstruiert, um Inkompatibilitäten bei Modifikationen zu vermeiden.
* **Cosine Annealing Lernraten-Scheduler**: Verringert die Lernrate kontinuierlich, um dem Modell in der Spätphase des Trainings extrem präzise Justierungen zu ermöglichen und ein Oszillieren oder Verfehlen des Balls zu verhindern.
* **Spiegelsymmetrische KI-Simulation**: Ermöglicht dem trainierten Modell im AI-vs-AI-Modus, fehlerfrei gegen sich selbst zu spielen, indem der Zustand des linken Schlägers für die Entscheidungsfindung mathematisch gespiegelt wird.

---

## Projektstruktur

* `dqn.py`: Definition des neuronalen Netzwerks (Dueling Q-Network) und automatische Geräteauswahl (CUDA, MPS, CPU).
* `pong_env.py`: Die physikalische Simulation des Pong-Spiels, sowohl als Einzelumgebung als auch als vektorisierte Multi-Umgebung (`VecPongEnv`).
* `train_dqn.py`: Das performante Trainingsskript mit GPU-Puffer, AMP (Mixed Precision) und hardwareoptimierter Thread-Steuerung.
* `play_pong.py`: Startet eine Benutzeroberfläche, in der ein menschlicher Spieler (Tastatur: W/S) gegen das trainierte Modell antritt.
* `play_pong_ai.py`: Demonstration, in der das Modell über Spiegelungsmatrizen gegen ein Duplikat seiner selbst spielt.

---

## Installation & Setup

### Voraussetzungen
Stellen Sie sicher, dass Python (3.10 oder neuer) sowie die passenden Treiber für Ihre GPU installiert sind (NVIDIA CUDA oder AMD ROCm).

### 1. Repository klonen
```bash
git clone https://github.com/IHR_USERNAME/vectorized-dqn-pong.git
cd vectorized-dqn-pong
```

### 2. Virtuelle Umgebung erstellen und aktivieren
```bash
python -m venv .venv
source .venv/bin/bin/activate  # Unter Windows: .venv\Scripts\activate
```

### 3. Abhängigkeiten installieren
```bash
pip install torch pygame numpy tensorboard
```

---

## Hardware-Anpassung & Skalierung

Die Parameter in `train_dqn.py` sind standardmäßig für High-End-Hardware optimiert. Sie können die Leistungsauslastung gezielt an Ihre Systemkomponenten anpassen.

### CPU-Threadsteuerung (Ryzen / Core CPUs)
In `train_dqn.py` im Einstieg der Funktion `train()`:
```python
torch.set_num_threads(32)          # Entspricht den logischen Threads Ihrer CPU (z.B. 32 bei Ryzen 9 9950X)
torch.set_num_interop_threads(4)   # Steuert die Inter-OP-Parallelisierung
```
* **Anpassung**: Setzen Sie `set_num_threads` auf die genaue Thread-Anzahl Ihres Prozessors, um den Rechenaufwand der physikalischen Umgebungen optimal zu verteilen.

### GPU-Sättigung und Speicherbandbreite (NVIDIA RTX / AMD Radeon)
* `num_envs` (Standard: `1024`): Anzahl parallel laufender Umgebungen. Höhere Werte belasten die CPU stärker, füllen aber den Replay-Buffer schneller.
* `batch_size` (Standard: `4096`): Die Anzahl der Samples pro Gradientenschritt.
  * **High-End GPUs (z. B. RTX 4080/4090, RX 7900 XT)**: Nutzen Sie `4096` oder `8192`.
  * **Mid-Range GPUs (z. B. RTX 3060, RX 6700 XT)**: Nutzen Sie `2048`.
  * **Low-End GPUs / CPU-only**: Nutzen Sie `512` oder `1024`.
* `updates_per_step` (Standard: `8`): Wie oft das Netzwerk pro Schritt trainiert wird. Ein höherer Wert erhöht die GPU-Auslastung drastisch. Falls das Modell zu früh overfittet, reduzieren Sie diesen Wert auf `2` oder `4`.

---

## Ausführung

### 1. Training starten
Verwenden Sie TensorBoard im Hintergrund, um den Lernfortschritt (Reward-Entwicklung, Epsilon, Loss) visuell zu überwachen:
```bash
# In einem separaten Terminal starten:
tensorboard --logdir=runs

# Trainingsprozess starten:
python train_dqn.py
```
Das Skript speichert bei neuen Bestwerten automatisch die Datei `best_pong_ai.pth` und periodisch die Datei `pong_ai.pth` ab.

### 2. Gegen die KI spielen
Sobald ein Modell trainiert wurde, können Sie Ihre Spielstärke testen:
```bash
python play_pong.py
```
* **Steuerung**: Taste `W` (Schläger nach oben), Taste `S` (Schläger nach unten).

### 3. KI gegen sich selbst spielen lassen
Um die gelernte Strategie im direkten Wettstreit zweier symmetrischer Agenten zu begutachten:
```bash
python play_pong_ai.py
```

---

## Technische Details des Reinforcement Learnings

Das System löst typische Probleme des parallelisierten Deep-Q-Learnings durch gezieltes Reward-Engineering und Hyperparameter-Sparsicherheit:

* **Episoden-basierter Decay**: Die Explorationsrate (`Epsilon`) zerfällt basierend auf abgeschlossenen Episoden anstatt auf globalen Schritten. Dies verhindert, dass die Lernkurve bei einer Skalierung der parallelen Instanzen (`num_envs`) kollabiert.
* **Reward Shaping**:
  * Ballkontakt (`last_hit_right`): `+1.0` (Starkes positives Feedback zur Ballabwehr).
  * Ball verpasst (`last_miss_right`): `-0.5` (Bestrafung für Punktverlust).
  * Kontinuierliches Tracking: Ein minimaler, entfernungsabhängiger Bonus (`0.005 * (1.0 - Abstand)`), solange der Ball sich auf die KI zubewegt. Dies zwingt das Modell, sich frühzeitig in die Flugbahn einzureihen.
  * Anti-Jitter-Strafe: Bestraft unästhetische Richtungswechsel innerhalb aufeinanderfolgender Frames, um eine flüssige Bewegung der Schläger zu erzwingen.

---

## Lizenz

Dieses Projekt ist unter den Bedingungen der MIT-Lizenz lizenziert. Weitere Details finden Sie in der `LICENSE`-Datei.
