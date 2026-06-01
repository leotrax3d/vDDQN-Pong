import os
import json
import time
from datetime import datetime
import numpy as np
import pygame
import torch

from dqn import DQN, get_device
from pong_env import PongEnv


# --- RETRO ZEICHENSATZ FÜR PYTHON 3.14 FALLBACK ---
# Jeder Buchstabe wird auf einem 3x5 Grid definiert (3 Spalten, 5 Zeilen)
FONT_DATA = {
    '0': ["XXX", "X.X", "X.X", "X.X", "XXX"],
    '1': ["..X", "..X", "..X", "..X", "..X"],
    '2': ["XXX", "..X", "XXX", "X..", "XXX"],
    '3': ["XXX", "..X", "XXX", "..X", "XXX"],
    '4': ["X.X", "X.X", "XXX", "..X", "..X"],
    '5': ["XXX", "X..", "XXX", "..X", "XXX"],
    '6': ["XXX", "X..", "XXX", "X.X", "XXX"],
    '7': ["XXX", "..X", "..X", "..X", "..X"],
    '8': ["XXX", "X.X", "XXX", "X.X", "XXX"],
    '9': ["XXX", "X.X", "XXX", "..X", "XXX"],
    'A': ["XXX", "X.X", "XXX", "X.X", "X.X"],
    'B': ["XX.", "X.X", "XX.", "X.X", "XX."],
    'C': ["XXX", "X..", "X..", "X..", "XXX"],
    'D': ["XX.", "X.X", "X.X", "X.X", "XX."],
    'E': ["XXX", "X..", "XXX", "X..", "XXX"],
    'F': ["XXX", "X..", "XXX", "X..", "X.."],
    'G': ["XXX", "X..", "X.X", "X.X", "XXX"],
    'H': ["X.X", "X.X", "XXX", "X.X", "X.X"],
    'I': ["XXX", ".X.", ".X.", ".X.", "XXX"],
    'J': ["..X", "..X", "..X", "X.X", "XXX"],
    'K': ["X.X", "X.X", "XX.", "X.X", "X.X"],
    'L': ["X..", "X..", "X..", "X..", "XXX"],
    'M': ["X.X", "XXX", "X.X", "X.X", "X.X"],
    'N': ["X.X", "XXX", "X.X", "X.X", "X.X"],
    'O': ["XXX", "X.X", "X.X", "X.X", "XXX"],
    'P': ["XXX", "X.X", "XXX", "X..", "X.."],
    'Q': ["XXX", "X.X", "XXX", "..X", "..X"],
    'R': ["XXX", "X.X", "XX.", "X.X", "X.X"],
    'S': ["XXX", "X..", "XXX", "..X", "XXX"],
    'T': ["XXX", ".X.", ".X.", ".X.", ".X."],
    'U': ["X.X", "X.X", "X.X", "X.X", "XXX"],
    'V': ["X.X", "X.X", "X.X", "X.X", ".X."],
    'W': ["X.X", "X.X", "X.X", "X.X", "XXX"],
    'X': ["X.X", "X.X", ".X.", "X.X", "X.X"],
    'Y': ["X.X", "X.X", ".X.", ".X.", ".X."],
    'Z': ["XXX", "..X", ".X.", "X..", "XXX"],
    ' ': ["...", "...", "...", "...", "..."],
    '%': ["X.X", "..X", ".X.", "X..", "X.X"],
    '.': ["...", "...", "...", "...", "..X"],
    ':': ["...", ".X.", "...", ".X.", "..."],
    '-': ["...", "...", "XXX", "...", "..."],
    '(': [".X.", "X..", "X..", "X..", ".X."],
    ')': [".X.", "..X", "..X", "..X", ".X."],
    ',': ["...", "...", "...", "..X", ".X."],
    '/': ["..X", ".X.", ".X.", ".X.", "X.."]
}


def draw_custom_char(surface, char, x, y, size, color):
    char = char.upper()
    if char not in FONT_DATA:
        char = ' '
    grid = FONT_DATA[char]
    for r_idx, row in enumerate(grid):
        for c_idx, val in enumerate(row):
            if val == 'X':
                pygame.draw.rect(
                    surface, 
                    color, 
                    (x + c_idx * size, y + r_idx * size, size, size)
                )


def draw_custom_text(surface, text, x, y, size=1, color=(255, 255, 255), align_center=False):
    text = str(text).upper()
    char_width = 3 * size
    char_spacing = 1 * size
    total_width = len(text) * char_width + (len(text) - 1) * char_spacing
    
    start_x = x - total_width // 2 if align_center else x
    
    for i, char in enumerate(text):
        draw_custom_char(surface, char, start_x + i * (char_width + char_spacing), y, size, color)


# Dynamische Auswahl zwischen nativem Font und Retro-Fallback
FONT_AVAILABLE = False
font_main = None
font_large = None


def draw_text(surface, text, x, y, color=(255, 255, 255), align_center=False, use_large=False):
    global FONT_AVAILABLE, font_main, font_large
    if FONT_AVAILABLE:
        font = font_large if use_large else font_main
        text_surf = font.render(text, True, color)
        rect = text_surf.get_rect()
        if align_center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        surface.blit(text_surf, rect)
    else:
        # Skaliert Retro-Rendering
        size = 4 if use_large else 2
        draw_custom_text(surface, text, x, y, size=size, color=color, align_center=align_center)


def update_paddle(env, action, paddle_num):
    speed = env.PADDLE_SPEED
    if paddle_num == 1:
        if action == PongEnv.ACTION_UP:
            env.paddle1_y -= speed
        elif action == PongEnv.ACTION_DOWN:
            env.paddle1_y += speed
        env.paddle1_y = env._clamp_paddle(env.paddle1_y)
    elif paddle_num == 2:
        if action == PongEnv.ACTION_UP:
            env.paddle2_y -= speed
        elif action == PongEnv.ACTION_DOWN:
            env.paddle2_y += speed
        env.paddle2_y = env._clamp_paddle(env.paddle2_y)


def get_model_decision(model, state, device):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t).squeeze(0).cpu().numpy()
        action = int(np.argmax(q_values))
        
        # Softmax zur Normalisierung
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()
        return action, probs


def get_mirrored_state(state):
    return [
        1.0 - state[0],
        state[1],
        -state[2],
        state[3],
        state[5],
        state[4]
    ]


def draw_q_bars(surface, x_start, y_start, probs, labels, title):
    width = 120
    bar_height = 12
    spacing = 18
    
    # Hintergrund-Box
    box_rect = pygame.Rect(x_start - 10, y_start - 20, width + 65, 80)
    pygame.draw.rect(surface, (45, 45, 60), box_rect, border_radius=5)
    
    # Titel
    draw_text(surface, title, x_start, y_start - 15, color=(203, 166, 247))
    
    for i, (prob, label) in enumerate(zip(probs, labels)):
        y = y_start + 10 + i * spacing
        # Label
        draw_text(surface, label, x_start, y, color=(166, 173, 200))
        
        # Leerer Hintergrundbalken
        pygame.draw.rect(surface, (90, 90, 110), (x_start + 45, y + 2, width, bar_height), border_radius=3)
        # Gefüllter Prozentbalken
        fill_width = int(width * prob)
        if fill_width > 0:
            pygame.draw.rect(surface, (243, 139, 168), (x_start + 45, y + 2, fill_width, bar_height), border_radius=3)
        
        # Prozentwert
        draw_text(surface, f"{int(prob*100)}%", x_start + 50 + width, y, color=(255, 255, 255))


def main():
    global FONT_AVAILABLE, font_main, font_large

    weights_path = "best_pong_ai.pth" if os.path.exists("best_pong_ai.pth") else "pong_ai.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Keine Modelldatei unter '{weights_path}' gefunden. "
            "Bitte trainieren Sie zuerst das Modell."
        )

    device = get_device()
    
    state_dict = torch.load(weights_path, map_location=device)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    hidden_size = 512
    value_hidden_size = None
    adv_hidden_size = None

    if "feature.0.bias" in clean_state_dict:
        hidden_size = clean_state_dict["feature.0.bias"].shape[0]
    if "value_stream.0.bias" in clean_state_dict:
        value_hidden_size = clean_state_dict["value_stream.0.bias"].shape[0]
    if "adv_stream.0.bias" in clean_state_dict:
        adv_hidden_size = clean_state_dict["adv_stream.0.bias"].shape[0]

    model = DQN(
        hidden_size=hidden_size,
        value_hidden_size=value_hidden_size,
        adv_hidden_size=adv_hidden_size
    ).to(device)
    model.load_state_dict(clean_state_dict)
    model.eval()

    pygame.display.init()
    
    # Sicherer Import-Versuch für Fonts (Python 3.14 Schutz)
    try:
        pygame.font.init()
        font_main = pygame.font.SysFont("Consolas", 12)
        font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        FONT_AVAILABLE = True
    except (ImportError, NotImplementedError, pygame.error):
        print("pygame.font ist auf diesem System nicht verfügbar. Verwende retro-vektorierten Zeichensatz.")

    window_width = 640
    window_height = 460
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Pong AI vs AI Cockpit")

    env = PongEnv(ai_paddle=False)
    clock = pygame.time.Clock()

    start_time = time.time()
    left_hits = 0
    right_hits = 0
    left_misses = 0
    right_misses = 0
    current_rally_hits = 0
    rally_lengths = []
    
    probs_left = np.array([0.33, 0.33, 0.34])
    probs_right = np.array([0.33, 0.33, 0.34])

    print(f"KI vs. KI Modus gestartet! Metriken werden erfasst. Modell: {weights_path}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = env._get_state()

        # Rechter Schläger
        action_right, probs_right = get_model_decision(model, state, device)
        update_paddle(env, action_right, paddle_num=2)

        # Linker Schläger (Spiegelung)
        mirrored_state = get_mirrored_state(state)
        action_left, probs_left = get_model_decision(model, mirrored_state, device)
        update_paddle(env, action_left, paddle_num=1)

        previous_vx = env.ball_vx
        _, _, done = env.step(PongEnv.ACTION_STAY)
        
        if done:
            if env.ball_x + env.BALL_SIZE < 0:
                left_misses += 1
            elif env.ball_x > env.FIELD_WIDTH:
                right_misses += 1
            
            rally_lengths.append(current_rally_hits)
            current_rally_hits = 0
            env.reset()
        else:
            if previous_vx < 0 and env.ball_vx > 0:
                left_hits += 1
                current_rally_hits += 1
            elif previous_vx > 0 and env.ball_vx < 0:
                right_hits += 1
                current_rally_hits += 1

        env.render()
        screen.blit(env._screen, (0, 0))

        # Cockpit Trennlinie
        pygame.draw.line(screen, (79, 79, 99), (0, 360), (window_width, 360), 2)
        pygame.draw.rect(screen, (30, 30, 46), (0, 361, window_width, 100))

        # Berechnungen der Fehlerraten
        left_attempts = left_hits + left_misses
        right_attempts = right_hits + right_misses
        left_rate = (left_misses / left_attempts * 100) if left_attempts > 0 else 0.0
        right_rate = (right_misses / right_attempts * 100) if right_attempts > 0 else 0.0
        
        # Punkteanzeige auf dem Feld
        draw_text(screen, f"{right_misses}   {left_misses}", window_width // 2, 40, color=(147, 153, 178), align_center=True, use_large=True)

        # Box LINKS
        pygame.draw.rect(screen, (45, 45, 60, 100), (10, 10, 155, 55), border_radius=4)
        draw_text(screen, "KI LINKS (GESPIEGELT)", 15, 15, color=(203, 166, 247))
        draw_text(screen, f"MISS: {left_rate:.1f}%", 15, 28, color=(243, 139, 168))
        draw_text(screen, f"HITS: {left_hits}", 15, 41, color=(166, 173, 200))

        # Box RECHTS
        pygame.draw.rect(screen, (45, 45, 60, 100), (window_width - 165, 10, 155, 55), border_radius=4)
        draw_text(screen, "KI RECHTS (NORMAL)", window_width - 160, 15, color=(203, 166, 247))
        draw_text(screen, f"MISS: {right_rate:.1f}%", window_width - 160, 28, color=(243, 139, 168))
        draw_text(screen, f"HITS: {right_hits}", window_width - 160, 41, color=(166, 173, 200))

        # Mittlerer Text
        draw_text(screen, f"STREAK: {current_rally_hits}", window_width // 2, 15, color=(249, 226, 175), align_center=True)

        # Q-Wert-Balken
        labels = ["STAY", "UP", "DOWN"]
        draw_q_bars(screen, 30, 385, probs_left, labels, "ZUSTANDS-VERTRAUEN LINKS")
        draw_q_bars(screen, window_width - 215, 385, probs_right, labels, "ZUSTANDS-VERTRAUEN RECHTS")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    end_time = time.time()
    duration = end_time - start_time
    if current_rally_hits > 0:
        rally_lengths.append(current_rally_hits)
    total_rallies = len(rally_lengths)
    total_hits = left_hits + right_hits
    left_attempts = left_hits + left_misses
    right_attempts = right_hits + right_misses
    left_miss_rate = (left_misses / left_attempts * 100) if left_attempts > 0 else 0.0
    right_miss_rate = (right_misses / right_attempts * 100) if right_attempts > 0 else 0.0
    max_rally = max(rally_lengths) if total_rallies > 0 else 0
    avg_rally = (sum(rally_lengths) / total_rallies) if total_rallies > 0 else 0.0

    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_duration_seconds": round(duration, 2),
        "total_rallies": total_rallies,
        "left_hits": left_hits,
        "left_misses": left_misses,
        "right_hits": right_hits,
        "right_misses": right_misses,
        "left_miss_rate_percent": round(left_miss_rate, 2),
        "right_miss_rate_percent": round(right_miss_rate, 2),
        "max_rally_length": max_rally,
        "average_rally_length": round(avg_rally, 2)
    }

    with open("pong_ai_session_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    print("\n" + "="*50)
    print("             SESSION STATISTIKEN KI vs. KI            ")
    print("="*50)
    print(f"Spieldauer:             {duration:.2f} Sekunden")
    print(f"Ballwechsel gesamt:     {total_rallies}")
    print(f"Ballkontakte gesamt:    {total_hits}")
    print("-"*50)
    print(f"Fehlerrate LINKS (Miss): {left_miss_rate:.2f}% ({left_misses} von {left_attempts} Bällen)")
    print(f"Fehlerrate RECHTS (Miss):{right_miss_rate:.2f}% ({right_misses} von {right_attempts} Bällen)")
    print("-"*50)
    print(f"Längster Ballwechsel:   {max_rally} Ballkontakte")
    print(f"Schnitt pro Ballwechsel:{avg_rally:.2f} Ballkontakte")
    print("="*50)
    print("Statistiken erfolgreich in 'pong_ai_session_stats.json' gespeichert.\n")


if __name__ == "__main__":
    main()