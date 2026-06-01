import math
import random
import numpy as np
import pygame


class PongEnv:
    FIELD_WIDTH = 640
    FIELD_HEIGHT = 360
    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 60
    PADDLE_SPEED = 6
    BALL_SIZE = 8
    BALL_SPEED_X = 4
    BALL_SPEED_Y = 3
    MAX_BALL_SPEED_Y = 6
    MAX_BOUNCE_ANGLE_DEG = 60

    COLOR_BG = (30, 30, 46)
    COLOR_PADDLE = (203, 166, 247)
    COLOR_BALL = (243, 139, 168)

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2

    def __init__(self, seed=None, ai_paddle=True):
        self.rng = random.Random(seed)
        self.ai_paddle = ai_paddle
        self._screen = None
        self.reset()

    def reset(self):
        self.ball_x = self.FIELD_WIDTH // 2
        self.ball_y = self.FIELD_HEIGHT // 2

        dir_x = self.rng.choice([-1, 1])
        dir_y = self.rng.choice([-1, 1])
        self.ball_vx = dir_x * self.BALL_SPEED_X
        self.ball_vy = dir_y * self.BALL_SPEED_Y

        self.paddle1_y = (self.FIELD_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.paddle2_y = (self.FIELD_HEIGHT - self.PADDLE_HEIGHT) // 2

        self.last_hit_right = False
        self.last_miss_right = False

        return self._get_state()

    def step(self, action):
        self.last_hit_right = False
        self.last_miss_right = False

        # Menschlicher Schläger (Links)
        if action == self.ACTION_UP:
            self.paddle1_y -= self.PADDLE_SPEED
        elif action == self.ACTION_DOWN:
            self.paddle1_y += self.PADDLE_SPEED
        self.paddle1_y = self._clamp_paddle(self.paddle1_y)

        # Bot-Schläger (Rechts)
        if self.ai_paddle:
            paddle2_center = self.paddle2_y + self.PADDLE_HEIGHT // 2
            if self.ball_y < paddle2_center:
                self.paddle2_y -= self.PADDLE_SPEED
            elif self.ball_y > paddle2_center:
                self.paddle2_y += self.PADDLE_SPEED
            self.paddle2_y = self._clamp_paddle(self.paddle2_y)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wandkollisionen (Oben/Unten)
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = -self.ball_vy
        elif self.ball_y + self.BALL_SIZE >= self.FIELD_HEIGHT:
            self.ball_y = self.FIELD_HEIGHT - self.BALL_SIZE
            self.ball_vy = -self.ball_vy

        # Schläger-Kollisionen
        left_paddle_x = 0
        right_paddle_x = self.FIELD_WIDTH - self.PADDLE_WIDTH

        if self._check_paddle_collision(left_paddle_x, self.paddle1_y):
            self.ball_x = left_paddle_x + self.PADDLE_WIDTH
            self._apply_spin(self.paddle1_y, direction=1)
        elif self._check_paddle_collision(right_paddle_x, self.paddle2_y):
            self.ball_x = right_paddle_x - self.BALL_SIZE
            self._apply_spin(self.paddle2_y, direction=-1)
            self.last_hit_right = True

        reward = 0
        done = False
        if self.ball_x + self.BALL_SIZE < 0:
            reward = -1
            done = True
        elif self.ball_x > self.FIELD_WIDTH:
            reward = 1
            done = True
            self.last_miss_right = True

        return self._get_state(), reward, done

    def render(self):
        if self._screen is None:
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.FIELD_WIDTH, self.FIELD_HEIGHT))
            pygame.display.set_caption("Pong")

        self._screen.fill(self.COLOR_BG)

        left_paddle_rect = pygame.Rect(0, self.paddle1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        right_paddle_rect = pygame.Rect(
            self.FIELD_WIDTH - self.PADDLE_WIDTH,
            self.paddle2_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        pygame.draw.rect(self._screen, self.COLOR_PADDLE, left_paddle_rect)
        pygame.draw.rect(self._screen, self.COLOR_PADDLE, right_paddle_rect)
        pygame.draw.rect(self._screen, self.COLOR_BALL, ball_rect)

        pygame.display.flip()

    def _get_state(self):
        max_ball_speed = math.hypot(self.BALL_SPEED_X, self.MAX_BALL_SPEED_Y)
        denom_x = max(1, self.FIELD_WIDTH - self.BALL_SIZE)
        denom_y = max(1, self.FIELD_HEIGHT - self.BALL_SIZE)
        denom_paddle = max(1, self.FIELD_HEIGHT - self.PADDLE_HEIGHT)

        return [
            self.ball_x / denom_x,
            self.ball_y / denom_y,
            self.ball_vx / max_ball_speed,
            self.ball_vy / max_ball_speed,
            self.paddle1_y / denom_paddle,
            self.paddle2_y / denom_paddle,
        ]

    def _clamp_paddle(self, paddle_y):
        return max(0, min(self.FIELD_HEIGHT - self.PADDLE_HEIGHT, paddle_y))

    def _check_paddle_collision(self, paddle_x, paddle_y):
        ball_right = self.ball_x + self.BALL_SIZE
        ball_bottom = self.ball_y + self.BALL_SIZE

        overlap_x = (self.ball_x < paddle_x + self.PADDLE_WIDTH) and (ball_right > paddle_x)
        overlap_y = (self.ball_y < paddle_y + self.PADDLE_HEIGHT) and (ball_bottom > paddle_y)
        return overlap_x and overlap_y

    def _apply_spin(self, paddle_y, direction):
        paddle_center = paddle_y + self.PADDLE_HEIGHT / 2
        ball_center = self.ball_y + self.BALL_SIZE / 2
        offset = (ball_center - paddle_center) / (self.PADDLE_HEIGHT / 2)
        offset = max(-1.0, min(1.0, offset))

        speed = math.hypot(self.ball_vx, self.ball_vy)
        angle = math.radians(self.MAX_BOUNCE_ANGLE_DEG) * offset

        vx = direction * speed * math.cos(angle)
        vy = speed * math.sin(angle)

        new_vx = int(round(vx))
        if new_vx == 0:
            new_vx = direction
        self.ball_vx = new_vx
        self.ball_vy = int(round(vy))
        if abs(self.ball_vy) > self.MAX_BALL_SPEED_Y:
            self.ball_vy = int(math.copysign(self.MAX_BALL_SPEED_Y, self.ball_vy))


class VecPongEnv:
    """Komplette Vektorumgebung, damit train_dqn.py uneingeschränkt funktioniert."""
    def __init__(self, num_envs, seed=None):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)

        self.ball_x = np.zeros(num_envs, dtype=np.float32)
        self.ball_y = np.zeros(num_envs, dtype=np.float32)
        self.ball_vx = np.zeros(num_envs, dtype=np.float32)
        self.ball_vy = np.zeros(num_envs, dtype=np.float32)
        self.paddle1_y = np.zeros(num_envs, dtype=np.float32)
        self.paddle2_y = np.zeros(num_envs, dtype=np.float32)

        self.last_hit_right = np.zeros(num_envs, dtype=bool)
        self.last_miss_right = np.zeros(num_envs, dtype=bool)

        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        mask_idx = np.where(mask)[0]
        if mask_idx.size == 0:
            return self._get_state()

        self.ball_x[mask_idx] = PongEnv.FIELD_WIDTH // 2
        self.ball_y[mask_idx] = PongEnv.FIELD_HEIGHT // 2

        dir_x = self.rng.choice([-1, 1], size=mask_idx.size)
        dir_y = self.rng.choice([-1, 1], size=mask_idx.size)
        self.ball_vx[mask_idx] = dir_x * PongEnv.BALL_SPEED_X
        self.ball_vy[mask_idx] = dir_y * PongEnv.BALL_SPEED_Y

        center_y = (PongEnv.FIELD_HEIGHT - PongEnv.PADDLE_HEIGHT) // 2
        self.paddle1_y[mask_idx] = center_y
        self.paddle2_y[mask_idx] = center_y

        self.last_hit_right[mask_idx] = False
        self.last_miss_right[mask_idx] = False

        return self._get_state()

    def left_dummy_actions(self):
        paddle_center = self.paddle1_y + PongEnv.PADDLE_HEIGHT / 2
        actions = np.full(self.num_envs, PongEnv.ACTION_STAY, dtype=np.int64)
        actions[self.ball_y < paddle_center] = PongEnv.ACTION_UP
        actions[self.ball_y > paddle_center] = PongEnv.ACTION_DOWN
        return actions

    def step(self, action_left, action_right):
        self.last_hit_right.fill(False)
        self.last_miss_right.fill(False)

        self.paddle1_y[action_left == PongEnv.ACTION_UP] -= PongEnv.PADDLE_SPEED
        self.paddle1_y[action_left == PongEnv.ACTION_DOWN] += PongEnv.PADDLE_SPEED

        self.paddle2_y[action_right == PongEnv.ACTION_UP] -= PongEnv.PADDLE_SPEED
        self.paddle2_y[action_right == PongEnv.ACTION_DOWN] += PongEnv.PADDLE_SPEED

        max_paddle_y = PongEnv.FIELD_HEIGHT - PongEnv.PADDLE_HEIGHT
        self.paddle1_y = np.clip(self.paddle1_y, 0, max_paddle_y)
        self.paddle2_y = np.clip(self.paddle2_y, 0, max_paddle_y)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        hit_top = self.ball_y <= 0
        hit_bottom = self.ball_y + PongEnv.BALL_SIZE >= PongEnv.FIELD_HEIGHT
        self.ball_y[hit_top] = 0
        self.ball_y[hit_bottom] = PongEnv.FIELD_HEIGHT - PongEnv.BALL_SIZE
        self.ball_vy[hit_top | hit_bottom] *= -1

        left_paddle_x = 0
        right_paddle_x = PongEnv.FIELD_WIDTH - PongEnv.PADDLE_WIDTH
        ball_right = self.ball_x + PongEnv.BALL_SIZE
        ball_bottom = self.ball_y + PongEnv.BALL_SIZE

        left_overlap_x = (self.ball_x < left_paddle_x + PongEnv.PADDLE_WIDTH) & (ball_right > left_paddle_x)
        left_overlap_y = (self.ball_y < self.paddle1_y + PongEnv.PADDLE_HEIGHT) & (ball_bottom > self.paddle1_y)
        hit_left = left_overlap_x & left_overlap_y

        right_overlap_x = (self.ball_x < right_paddle_x + PongEnv.PADDLE_WIDTH) & (ball_right > right_paddle_x)
        right_overlap_y = (self.ball_y < self.paddle2_y + PongEnv.PADDLE_HEIGHT) & (ball_bottom > self.paddle2_y)
        hit_right = right_overlap_x & right_overlap_y

        if np.any(hit_left):
            self.ball_x[hit_left] = left_paddle_x + PongEnv.PADDLE_WIDTH
            self._apply_spin(hit_left, direction=1)

        if np.any(hit_right):
            self.ball_x[hit_right] = right_paddle_x - PongEnv.BALL_SIZE
            self._apply_spin(hit_right, direction=-1)
            self.last_hit_right[hit_right] = True

        reward_left = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)

        out_left = self.ball_x + PongEnv.BALL_SIZE < 0
        out_right = self.ball_x > PongEnv.FIELD_WIDTH

        reward_left[out_left] = -1.0
        reward_left[out_right] = 1.0
        done[out_left | out_right] = True
        self.last_miss_right[out_right] = True

        return self._get_state(), reward_left, done

    def _apply_spin(self, mask, direction):
        if not np.any(mask):
            return
        
        paddle_y = self.paddle2_y[mask] if direction == -1 else self.paddle1_y[mask]
        paddle_center = paddle_y + PongEnv.PADDLE_HEIGHT / 2
        ball_center = self.ball_y[mask] + PongEnv.BALL_SIZE / 2
        offset = (ball_center - paddle_center) / (PongEnv.PADDLE_HEIGHT / 2)
        offset = np.clip(offset, -1.0, 1.0)

        speed = np.sqrt(self.ball_vx[mask] ** 2 + self.ball_vy[mask] ** 2)
        angle = np.deg2rad(PongEnv.MAX_BOUNCE_ANGLE_DEG) * offset

        vx = direction * speed * np.cos(angle)
        vy = speed * np.sin(angle)

        new_vx = np.rint(vx).astype(np.int32)
        new_vx[new_vx == 0] = direction
        self.ball_vx[mask] = new_vx
        self.ball_vy[mask] = np.rint(vy)

        max_vy = PongEnv.MAX_BALL_SPEED_Y
        self.ball_vy[mask] = np.clip(self.ball_vy[mask], -max_vy, max_vy)

    def _get_state(self):
        max_ball_speed = math.hypot(PongEnv.BALL_SPEED_X, PongEnv.MAX_BALL_SPEED_Y)
        denom_x = max(1, PongEnv.FIELD_WIDTH - PongEnv.BALL_SIZE)
        denom_y = max(1, PongEnv.FIELD_HEIGHT - PongEnv.BALL_SIZE)
        denom_paddle = max(1, PongEnv.FIELD_HEIGHT - PongEnv.PADDLE_HEIGHT)

        return np.stack(
            [
                self.ball_x / denom_x,
                self.ball_y / denom_y,
                self.ball_vx / max_ball_speed,
                self.ball_vy / max_ball_speed,
                self.paddle1_y / denom_paddle,
                self.paddle2_y / denom_paddle,
            ],
            axis=1,
        )