import os
import pygame
import torch

from dqn import DQN, get_device
from pong_env import PongEnv


def update_right_paddle(env, action):
    if action == PongEnv.ACTION_UP:
        env.paddle2_y -= env.PADDLE_SPEED
    elif action == PongEnv.ACTION_DOWN:
        env.paddle2_y += env.PADDLE_SPEED
    env.paddle2_y = env._clamp_paddle(env.paddle2_y)


def select_greedy_action(model, state, device):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def main():
    weights_path = "best_pong_ai.pth" if os.path.exists("best_pong_ai.pth") else "pong_ai.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Keine Modelldatei unter '{weights_path}' gefunden. "
            "Bitte trainieren Sie zuerst das Modell mit 'train_dqn.py'."
        )

    device = get_device()
    
    state_dict = torch.load(weights_path, map_location=device)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    # Automatische Erkennung der Layer-Dimensionen aus dem Checkpoint
    hidden_size = 512
    value_hidden_size = None
    adv_hidden_size = None

    if "feature.0.bias" in clean_state_dict:
        hidden_size = clean_state_dict["feature.0.bias"].shape[0]
    if "value_stream.0.bias" in clean_state_dict:
        value_hidden_size = clean_state_dict["value_stream.0.bias"].shape[0]
    if "adv_stream.0.bias" in clean_state_dict:
        adv_hidden_size = clean_state_dict["adv_stream.0.bias"].shape[0]

    # Modell mit exakt passenden Dimensionen initialisieren
    model = DQN(
        hidden_size=hidden_size,
        value_hidden_size=value_hidden_size,
        adv_hidden_size=adv_hidden_size
    ).to(device)
    model.load_state_dict(clean_state_dict)
    model.eval()

    pygame.display.init()
    
    env = PongEnv(ai_paddle=False)
    clock = pygame.time.Clock()

    print(f"Spiel gestartet! Steuerung: W (Hoch) | S (Runter). Gegner: {weights_path}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action_left = PongEnv.ACTION_STAY
        if keys[pygame.K_w]:
            action_left = PongEnv.ACTION_UP
        elif keys[pygame.K_s]:
            action_left = PongEnv.ACTION_DOWN

        action_right = select_greedy_action(model, env._get_state(), device)
        update_right_paddle(env, action_right)

        _, _, done = env.step(action_left)
        if done:
            env.reset()

        env.render()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()