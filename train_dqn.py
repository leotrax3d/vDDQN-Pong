import random
import numpy as np
from collections import deque

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN, get_device
from pong_env import PongEnv, VecPongEnv


class GPUReplayBuffer:
    """
    Ein vollständig auf der GPU residenter Replay-Buffer.
    Garantierte Datensicherheit durch synchronen Transfer.
    """
    def __init__(self, capacity, state_dim=6, device="cuda"):
        self.capacity = capacity
        self.device = device
        
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0

    def push_batch(self, states, actions, rewards, next_states, dones):
        n = len(states)
        assert n <= self.capacity, "Die Batch-Größe darf die Kapazität des Replay-Buffers nicht überschreiten."
        
        states_t = torch.from_numpy(states).to(self.device, dtype=torch.float32)
        actions_t = torch.from_numpy(actions).to(self.device, dtype=torch.int64)
        rewards_t = torch.from_numpy(rewards).to(self.device, dtype=torch.float32)
        next_states_t = torch.from_numpy(next_states).to(self.device, dtype=torch.float32)
        dones_t = torch.from_numpy(dones).to(self.device, dtype=torch.bool)

        if self.ptr + n <= self.capacity:
            self.states[self.ptr : self.ptr + n] = states_t
            self.actions[self.ptr : self.ptr + n] = actions_t
            self.rewards[self.ptr : self.ptr + n] = rewards_t
            self.next_states[self.ptr : self.ptr + n] = next_states_t
            self.dones[self.ptr : self.ptr + n] = dones_t
            self.ptr = (self.ptr + n) % self.capacity
        else:
            space = self.capacity - self.ptr
            self.states[self.ptr :] = states_t[:space]
            self.actions[self.ptr :] = actions_t[:space]
            self.rewards[self.ptr :] = rewards_t[:space]
            self.next_states[self.ptr :] = next_states_t[:space]
            self.dones[self.ptr :] = dones_t[:space]

            self.states[: n - space] = states_t[space:]
            self.actions[: n - space] = actions_t[space:]
            self.rewards[: n - space] = rewards_t[space:]
            self.next_states[: n - space] = next_states_t[space:]
            self.dones[: n - space] = dones_t[space:]
            self.ptr = n - space

        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


def compute_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    slope = (eps_end - eps_start) / eps_decay_steps
    return eps_start + slope * step


def compute_left_random_prob(step, prob_start, prob_end, decay_steps):
    if step >= decay_steps:
        return prob_end
    slope = (prob_end - prob_start) / decay_steps
    return prob_start + slope * step


def select_greedy_action(model, state, device):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def select_actions_batch(model, states, epsilon, device):
    states_t = torch.from_numpy(states).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        q_values = model(states_t)
        greedy_actions = torch.argmax(q_values, dim=1)

    batch_size = greedy_actions.shape[0]
    if epsilon <= 0.0:
        return greedy_actions.tolist()

    random_mask = torch.rand(batch_size, device=device) < epsilon
    random_actions = torch.randint(0, 3, (batch_size,), device=device)
    actions = torch.where(random_mask, random_actions, greedy_actions)
    return actions.tolist()


def left_dummy_action(env):
    paddle_center = env.paddle1_y + env.PADDLE_HEIGHT // 2
    if env.ball_y < paddle_center:
        return PongEnv.ACTION_UP
    if env.ball_y > paddle_center:
        return PongEnv.ACTION_DOWN
    return PongEnv.ACTION_STAY


def evaluate_agent(model, device, episodes, max_steps, opponent_policy):
    was_training = model.training
    model.eval()

    env = PongEnv(ai_paddle=False)
    wins = 0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action_right = select_greedy_action(model, state, device)
            if action_right == PongEnv.ACTION_UP:
                env.paddle2_y -= env.PADDLE_SPEED
            elif action_right == PongEnv.ACTION_DOWN:
                env.paddle2_y += env.PADDLE_SPEED
            env.paddle2_y = env._clamp_paddle(env.paddle2_y)

            action_left = opponent_policy(env)
            next_state, reward_left, done = env.step(action_left)
            state = next_state

            if done:
                if reward_left == -1:
                    wins += 1
                break

    if was_training:
        model.train()

    return wins / max(1, episodes)


def sync_target_net(policy_net, target_net):
    source = getattr(policy_net, "_orig_mod", policy_net)
    target_net.load_state_dict(source.state_dict())


def save_model(model, filepath):
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(raw_model.state_dict(), filepath)


def train():
    # --- CPU THREADPOOL OPTIMIERUNG ---
    torch.set_num_threads(32)
    torch.set_num_interop_threads(4)

    device = get_device()

    # --- HYPERPARAMETER ---
    max_gradient_steps = 100000    # Ziel: 100k echte Updates für perfektes, stabiles Lernen
    max_steps = 2000
    gamma = 0.99
    lr = 3e-4                      
    batch_size = 512               
    buffer_size = 1000000          
    
    warmup_steps = 100000          

    # Präziser linearer Zerfall über die Gradientenschritte
    eps_start = 1.0
    eps_end = 0.01                 
    eps_decay_steps = 60000        # Exploration läuft über 60k Updates
    
    target_update_steps = 1000     
    save_every_episodes = 500
    reward_window_size = 100
    
    left_random_start = 0.95
    left_random_end = 0.10
    left_random_decay_steps = 80000 # Erreicht maximale Stärke nach 80k Updates
    
    eval_every_episodes = 500
    eval_episodes = 50
    num_envs = 1024  
    updates_per_step = 8           
    # ---------------------------------------------------------

    envs = VecPongEnv(num_envs)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    sync_target_net(policy_net, target_net)
    
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-3)
    
    # Cosine Scheduler exakt synchronisiert auf 100k Updates
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_gradient_steps, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss()
    
    replay = GPUReplayBuffer(buffer_size, device=device.type)
    writer = SummaryWriter(log_dir="runs/pong_power_training")

    reward_window = deque(maxlen=reward_window_size)
    best_avg_reward = float("-inf")
    best_eval_winrate = float("-inf")

    previous_actions = np.zeros(num_envs, dtype=np.int64)
    gradient_steps = 0  
    completed_episodes = 0

    try:
        states = envs.reset()
        episode_rewards = np.zeros(num_envs, dtype=np.float32)
        steps_in_episode = np.zeros(num_envs, dtype=np.int32)
        loss_sum = 0.0
        loss_count = 0

        # Die Schleife läuft, bis die notwendigen 100k Updates vollständig abgeschlossen sind
        while gradient_steps < max_gradient_steps:
            is_warmed_up = len(replay) >= warmup_steps
            
            epsilon = compute_epsilon(gradient_steps, eps_start, eps_end, eps_decay_steps) if is_warmed_up else 1.0
            left_random_prob = compute_left_random_prob(gradient_steps, left_random_start, left_random_end, left_random_decay_steps) if is_warmed_up else left_random_start

            action_rights_list = select_actions_batch(policy_net, states, epsilon, device)
            action_rights = np.array(action_rights_list, dtype=np.int64)

            random_mask = np.random.rand(num_envs) < left_random_prob
            action_lefts = envs.left_dummy_actions()
            action_lefts[random_mask] = np.random.randint(0, 3, size=random_mask.sum())

            next_states, reward_lefts, dones = envs.step(action_lefts, action_rights)

            # --- BELOHNUNGSSYSTEM ---
            reward_rights = -reward_lefts  

            reward_rights[envs.last_hit_right] += 1.0
            reward_rights[envs.last_miss_right] -= 0.5

            ball_y = states[:, 1]
            paddle2_y = states[:, 5]

            # Physikalische Zentren fehlerfrei berechnen
            ball_y_pixel = ball_y * (PongEnv.FIELD_HEIGHT - PongEnv.BALL_SIZE) + PongEnv.BALL_SIZE / 2
            paddle2_y_pixel = paddle2_y * (PongEnv.FIELD_HEIGHT - PongEnv.PADDLE_HEIGHT) + PongEnv.PADDLE_HEIGHT / 2

            physical_distance = np.abs(ball_y_pixel - paddle2_y_pixel) / PongEnv.FIELD_HEIGHT

            # Proaktives Shadowing: Ständige Belohnung für perfekte Positionierung
            tracking_reward = 0.005 * (1.0 - physical_distance)
            reward_rights += tracking_reward
            # -------------------------------------------------------------

            replay.push_batch(states, action_rights, reward_rights, next_states, dones)

            episode_rewards += reward_rights
            steps_in_episode += 1
            previous_actions = action_rights.copy()

            finished = dones | (steps_in_episode >= max_steps)
            finished_idx = np.where(finished)[0]
            for idx in finished_idx:
                if is_warmed_up:
                    completed_episodes += 1
                    
                    if completed_episodes % 1000 == 0:
                        save_model(policy_net, f"pong_ai_ep_{completed_episodes}.pth")
                        print(f"Generationen-Checkpoint für Episode {completed_episodes} gesichert.")
                    
                    reward_window.append(float(episode_rewards[idx]))
                    avg_reward = sum(reward_window) / len(reward_window)
                    avg_loss = loss_sum / loss_count if loss_count > 0 else float("nan")

                    writer.add_scalar("Total Reward", float(episode_rewards[idx]), completed_episodes)
                    writer.add_scalar("Epsilon", epsilon, completed_episodes)
                    writer.add_scalar("Loss", avg_loss, completed_episodes)
                    writer.add_scalar("Left Random Prob", left_random_prob, completed_episodes)
                    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], completed_episodes)

                    if avg_reward > best_avg_reward and completed_episodes > 100:
                        best_avg_reward = avg_reward
                        save_model(policy_net, "best_pong_ai.pth")
                        print(f"Bester Schnitt: {avg_reward:.2f} | Modell gespeichert.")

                    if completed_episodes % eval_every_episodes == 0:
                        winrate_dummy = evaluate_agent(
                            policy_net,
                            device,
                            eval_episodes,
                            max_steps,
                            left_dummy_action,
                        )
                        winrate_random = evaluate_agent(
                            policy_net,
                            device,
                            eval_episodes,
                            max_steps,
                            lambda _env: random.randrange(3),
                        )
                        writer.add_scalar("Eval Winrate Dummy", winrate_dummy, completed_episodes)
                        writer.add_scalar("Eval Winrate Random", winrate_random, completed_episodes)

                        eval_winrate = (winrate_dummy + winrate_random) / 2.0
                        if eval_winrate > best_eval_winrate:
                            best_eval_winrate = eval_winrate
                            save_model(policy_net, "best_pong_ai.pth")
                            print(
                                f"Evaluierungshoch! Winrate: {eval_winrate:.2f} "
                                f"(Dummy={winrate_dummy:.2f}, Random={winrate_random:.2f})"
                            )

                    if completed_episodes % save_every_episodes == 0:
                        save_model(policy_net, "pong_ai.pth")

                    if completed_episodes % 100 == 0:
                        print(
                            f"Episode {completed_episodes} | "
                            f"Updates {gradient_steps}/{max_gradient_steps} | "
                            f"Schnitt={avg_reward:.2f} | "
                            f"Epsilon={epsilon:.3f} | Buffer={len(replay)}"
                        )

                episode_rewards[idx] = 0.0
                steps_in_episode[idx] = 0

            if finished_idx.size > 0:
                envs.reset(finished)
                next_states = envs._get_state()

            states = next_states

            if is_warmed_up:
                for _ in range(updates_per_step):
                    states_t, actions_t, rewards_t, next_states_t, dones_t = replay.sample(batch_size)
                    
                    actions_t = actions_t.unsqueeze(1)
                    rewards_t = rewards_t.unsqueeze(1)
                    dones_t = dones_t.unsqueeze(1).float()

                    q_values = policy_net(states_t).gather(1, actions_t)
                    with torch.no_grad():
                        next_actions = policy_net(next_states_t).argmax(dim=1, keepdim=True)
                        next_q_values = target_net(next_states_t).gather(1, next_actions)
                        target_q = rewards_t + (1.0 - dones_t) * gamma * next_q_values

                    loss = loss_fn(q_values, target_q)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()

                    loss_sum += float(loss.item())
                    loss_count += 1

                    # Scheduler und Synchronisation hängen jetzt direkt am globalen Update-Zähler
                    scheduler.step()
                    gradient_steps += 1
                    
                    if gradient_steps % target_update_steps == 0:
                        sync_target_net(policy_net, target_net)

                    if gradient_steps >= max_gradient_steps:
                        break
    finally:
        writer.close()


if __name__ == "__main__":
    train()