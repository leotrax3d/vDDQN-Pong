# Vectorized Dueling DQN Pong with GPU-Resident Replay Buffer

## EN:
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg)](https://pytorch.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.6%20%2B-green.svg)](https://www.pygame.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains an optimized implementation of a Dueling Deep Q-Network (DQN) trained to play the classic game of Pong. The training pipeline utilizes a fully vectorized NumPy environment combined with a GPU-resident replay buffer to minimize data transfer latencies and fully saturate modern multi-core CPUs and GPUs.


## Key Features

* **Vectorized Environment (`VecPongEnv`)**: Allows the simultaneous execution of hundreds or thousands of Pong instances in a single Python process using highly efficient NumPy vectorization.
* **GPU-Resident Replay Buffer**: Eliminates the PCIe bottleneck. All transition tuples are held, indexed, and sampled directly on GPU memory (CUDA/ROCm), significantly reducing host-to-device data transfer during gradient steps.
* **Dueling DQN Architecture**: Separates the estimation of state value and action advantages to ensure stable and consistent policy updates.
* **Automatic Dimension Detection**: When loading checkpoints, the model dynamically reconstructs the hidden layer shapes from the saved weight matrices, preventing shape mismatch errors when experimenting with network sizes.
* **Cosine Annealing Learning Rate Scheduler**: Gradually decays the learning rate to allow fine-tuned weight adjustments in the later stages of training, reducing policy oscillation and ensuring precise paddle placement.
* **Mirror-Symmetric AI Simulation**: Enables the trained model to play against itself in an AI-vs-AI demonstration by mathematically mirroring the game state representation for the left paddle.


## Project Structure

* `dqn.py`: Neural network definition (Dueling Q-Network) and automatic hardware device selection (CUDA, MPS, CPU).
* `pong_env.py`: Physics engine and game loop for both the single environment and the vectorized multi-environment (`VecPongEnv`).
* `train_dqn.py`: High-performance training script utilizing the GPU replay buffer, AMP (Automatic Mixed Precision), and optimized thread pool settings.
* `play_pong.py`: Starts an interactive GUI where a human player (Keyboard: W/S) plays against the trained model.
* `play_pong_ai.py`: A demonstration script where the model plays against an identical instance of itself using mirrored inputs.



## Installation & Setup

### Prerequisites
Ensure Python (3.10 or newer) and the appropriate GPU drivers (NVIDIA CUDA or AMD ROCm) are installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/vectorized-dqn-pong.git
cd vectorized-dqn-pong
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch pygame numpy tensorboard
```

---

## Hardware Adaptation & Scaling

The default configurations in `train_dqn.py` are optimized for high-end systems. You can adjust the following parameters to scale the workload to your specific hardware setup.

### CPU Thread Allocation (Ryzen / Core CPUs)
At the beginning of the `train()` function in `train_dqn.py`:
```python
torch.set_num_threads(32)          # Match the physical or logical threads of your CPU (e.g., 32 for Ryzen 9 9950X)
torch.set_num_interop_threads(4)   # Controls inter-op parallelization
```
* **Adaptation**: Adjust `set_num_threads` to your CPU's actual thread count to maximize the parallel generation of game frames.

### GPU Saturation (NVIDIA RTX / AMD Radeon)
* `num_envs` (Default: `1024`): The number of parallel environments run on the CPU. Higher values generate data faster but increase CPU RAM utilization.
* `batch_size` (Default: `4096`): The size of the training batch sampled during each SGD step.
  * **High-End GPUs (e.g., RTX 4080/4090, RX 7900 XT)**: Use `4096` or `8192`.
  * **Mid-Range GPUs (e.g., RTX 3060, RX 6700 XT)**: Use `2048`.
  * **Low-End GPUs / CPU-only**: Use `512` or `1024`.
* `updates_per_step` (Default: `8`): The number of SGD updates executed per environment step. Higher values increase GPU computing workloads. If you observe early overfitting, reduce this parameter to `2` or `4`.



## Usage

### 1. Start Training
You can monitor training metrics (total reward, epsilon, loss, learning rate) using TensorBoard:
```bash
# Start TensorBoard in a separate terminal:
tensorboard --logdir=runs

# Run the training script:
python train_dqn.py
```
The script automatically saves the model as `best_pong_ai.pth` whenever a new average reward milestone is reached, and periodically saves checkpoints as `pong_ai.pth`.

### 2. Play Against the AI
Test your skills against the trained neural network:
```bash
python play_pong.py
```
* **Controls**: Press `W` to move the left paddle up, `S` to move it down.

### 3. Watch AI vs. AI Match
Watch the model compete against an identical version of itself:
```bash
python play_pong_ai.py
```



## Reinforcement Learning & Reward Design

The pipeline addresses common scaling issues in parallel Deep Q-Networks through robust hyperparameter choices and target-oriented reward engineering:

* **Episode-Based Decay**: Epsilon (exploration probability) and the opponent's randomness decay based on completed episodes rather than global environment steps. This keeps exploration stable, regardless of how many parallel environments (`num_envs`) are executed.
* **Reward Shaping**:
  * Ball Contact (`last_hit_right`): `+1.0` (Strong positive reinforcement for successful ball deflection).
  * Ball Missed (`last_miss_right`): `-0.5` (Negative reinforcement for point loss).
  * Continuous Tracking: A small, distance-dependent step reward (`0.005 * (1.0 - distance)`) applied when the ball is moving towards the agent. This incentivizes the paddle to align with the ball's trajectory early.
  * Anti-Jitter Penalty: Penalizes rapid direction changes between consecutive frames to ensure a smooth, stable, and human-like movement style.



## License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.


## DE

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg)](https://pytorch.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.6%20%2B-green.svg)](https://www.pygame.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieses Repository enthält eine hochgradig optimierte Implementierung eines Dueling Deep Q-Networks (DQN), das darauf trainiert wird, das klassische Spiel Pong zu meistern. Die Trainingspipeline nutzt eine vollständig vektorisierte NumPy-Umgebung sowie einen GPU-residenten Replay-Buffer, um die Latenzzeiten beim Daten-Transfer zu minimieren und moderne Multi-Core-CPUs sowie GPUs maximal auszulasten.



## Hauptfeatures

* **Vektorisierte Umgebung (`VecPongEnv`)**: Erlaubt die gleichzeitige Ausführung von hunderten oder tausenden Pong-Instanzen in einem einzigen Python-Prozess mittels hocheffizienter NumPy-Vektorisierung.
* **GPU-Residenter Replay-Buffer**: Eliminiert den PCIe-Flaschenhals. Alle Zustandsübergänge werden direkt auf der GPU (CUDA/ROCm) gehalten, indiziert und gesampelt, wodurch der Datentransfer zwischen Host und Device drastisch reduziert wird.
* **Dueling DQN Architektur**: Trennt die Schätzung des Zustandswertes (State Value) und des Vorteils einzelner Aktionen (Advantage Stream) für stabileres Lernen.
* **Automatische Breiten-Erkennung**: Beim Laden von Checkpoints werden die Dimensionen der verborgenen Schichten automatisch aus den Gewichtungsmatrizen rekonstruiert, um Inkompatibilitäten bei Modifikationen zu vermeiden.
* **Cosine Annealing Lernraten-Scheduler**: Verringert die Lernrate kontinuierlich, um dem Modell in der Spätphase des Trainings extrem präzise Justierungen zu ermöglichen und ein Oszillieren oder Verfehlen des Balls zu verhindern.
* **Spiegelsymmetrische KI-Simulation**: Ermöglicht dem trainierten Modell im AI-vs-AI-Modus, fehlerfrei gegen sich selbst zu spielen, indem der Zustand des linken Schlägers für die Entscheidungsfindung mathematisch gespiegelt wird.



## Projektstruktur

* `dqn.py`: Definition des neuronalen Netzwerks (Dueling Q-Network) und automatische Geräteauswahl (CUDA, MPS, CPU).
* `pong_env.py`: Die physikalische Simulation des Pong-Spiels, sowohl als Einzelumgebung als auch als vektorisierte Multi-Umgebung (`VecPongEnv`).
* `train_dqn.py`: Das performante Trainingsskript mit GPU-Puffer, AMP (Mixed Precision) und hardwareoptimierter Thread-Steuerung.
* `play_pong.py`: Startet eine Benutzeroberfläche, in der ein menschlicher Spieler (Tastatur: W/S) gegen das trainierte Modell antritt.
* `play_pong_ai.py`: Demonstration, in der das Modell über Spiegelungsmatrizen gegen ein Duplikat seiner selbst spielt.



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



## Technische Details des Reinforcement Learnings

Das System löst typische Probleme des parallelisierten Deep-Q-Learnings durch gezieltes Reward-Engineering und Hyperparameter-Sparsicherheit:

* **Episoden-basierter Decay**: Die Explorationsrate (`Epsilon`) zerfällt basierend auf abgeschlossenen Episoden anstatt auf globalen Schritten. Dies verhindert, dass die Lernkurve bei einer Skalierung der parallelen Instanzen (`num_envs`) kollabiert.
* **Reward Shaping**:
  * Ballkontakt (`last_hit_right`): `+1.0` (Starkes positives Feedback zur Ballabwehr).
  * Ball verpasst (`last_miss_right`): `-0.5` (Bestrafung für Punktverlust).
  * Kontinuierliches Tracking: Ein minimaler, entfernungsabhängiger Bonus (`0.005 * (1.0 - Abstand)`), solange der Ball sich auf die KI zubewegt. Dies zwingt das Modell, sich frühzeitig in die Flugbahn einzureihen.
  * Anti-Jitter-Strafe: Bestraft unästhetische Richtungswechsel innerhalb aufeinanderfolgender Frames, um eine flüssige Bewegung der Schläger zu erzwingen.



## Lizenz

Dieses Projekt ist unter den Bedingungen der MIT-Lizenz lizenziert. Weitere Details finden Sie in der `LICENSE`-Datei.
