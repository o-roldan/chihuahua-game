import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from logic.collision_handler import ball_hits_target, character_hits_ball
from map.level_loader import LevelLoader
from entities.character import Character
from entities.ball import Ball
from entities.target import Target
from utils.constants import (
    BALL_MAX_SPEED,
    CHARACTER_JUMP_FORCE,
    CHARACTER_MAX_FALL_SPEED,
    CHARACTER_SPRINT_SPEED,
    COLORS,
    SCREEN_SIZE,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)


class ChihuahuaEnv(gym.Env):
    """
    Gymnasium environment wrapping the Pygame-based Chihuahua game.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # Fixed simulation timestep: physics advance 1/60 s per step regardless of
    # wall-clock time -> deterministic and as fast as the CPU allows.
    SIM_DT = 1.0 / metadata["render_fps"]

    # Randomized starts (training only; eval keeps the canonical start):
    # dog x, ball x ~ U(across the floor), ball velocity ~ U(-v, v) per axis.
    RANDOM_START_BALL_SPEED = 5.0  # 25% of BALL_MAX_SPEED

    def __init__(self, render_mode=None, random_start=False):
        super().__init__()
        self.random_start = random_start
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Action space: 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=SPRINT
        self.action_space = spaces.Discrete(5)

        # States space
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0,  # char_x
                    0,  # char_y
                    -CHARACTER_SPRINT_SPEED,  # char_vx
                    -CHARACTER_JUMP_FORCE,  # char_vy (jump impulse)
                    0,  # ball_x
                    0,  # ball_y
                    -BALL_MAX_SPEED,  # ball_vx
                    -BALL_MAX_SPEED,  # ball_vy
                    0,  # tracking state: jumping
                    0,  # tracking state: sprinting
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    SCREEN_WIDTH,
                    SCREEN_HEIGHT,
                    CHARACTER_SPRINT_SPEED,
                    CHARACTER_MAX_FALL_SPEED,  # char_vy (gravity cap)
                    SCREEN_WIDTH,
                    SCREEN_HEIGHT,
                    BALL_MAX_SPEED,
                    BALL_MAX_SPEED,
                    1,
                    1,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("Chihuahua RL Environment")
        else:
            self.screen = pygame.Surface(SCREEN_SIZE)

        self.level = LevelLoader.load("level_01.json")
        self.clock = pygame.time.Clock()

        start_x, start_y = self.level.start_pos
        ball_x, ball_y = self.level.ball_spawn
        target_x, target_y = self.level.target

        self.character = Character(self.screen, start_x, start_y)
        self.ball = Ball(ball_x, ball_y)
        self.target = Target(target_x, target_y)
        self.prev_ball_pos = np.array([ball_x, ball_y], dtype=np.float32)

        # Canonical start state: every reset restores exactly this snapshot
        # (previously animation timers leaked across episodes).
        self.character.reset()
        self.ball.reset()
        self.prev_ball_pos = np.array(self.ball.rect.center, dtype=np.float32)
        self._initial_state = self.get_state()

    # ---- Snapshot API: the simulator itself is the model ----
    # Surfaces are stored by reference (never mutated), everything else by value.
    def get_state(self):
        c, b = self.character, self.ball
        return {
            "char_rect": tuple(c.rect),
            "char_motion": (c.dx, c.dy, c.jumping, c.sprinting),
            "char_state": c.current_state.name,
            "char_image": c.image,
            "char_anim": {
                k: (s.current_delta, s.current_sprite)
                for k, s in c.states_dict.items()
                if hasattr(s, "current_delta")
            },
            "ball_rect": tuple(b.rect),
            "ball_velocity": tuple(b.velocity),
            "ball_on_ground": b.on_ground,
            "prev_ball_pos": self.prev_ball_pos.copy(),
        }

    def set_state(self, s):
        c, b = self.character, self.ball
        c.rect = pygame.Rect(s["char_rect"])
        c.dx, c.dy, c.jumping, c.sprinting = s["char_motion"]
        c.current_state = c.states_dict[s["char_state"]]
        c.image = s["char_image"]
        for k, (delta, sprite) in s["char_anim"].items():
            c.states_dict[k].current_delta = delta
            c.states_dict[k].current_sprite = sprite
        b.rect = pygame.Rect(s["ball_rect"])
        b.velocity = pygame.math.Vector2(s["ball_velocity"])
        b.on_ground = s["ball_on_ground"]
        self.prev_ball_pos = s["prev_ball_pos"].copy()

    def _randomize_start(self):
        c, b = self.character, self.ball
        rng = self.np_random
        c.rect.x = int(rng.integers(0, SCREEN_WIDTH - c.rect.width + 1))
        half = b.rect.width // 2
        b.rect.centerx = int(rng.integers(half, SCREEN_WIDTH - half + 1))
        v = self.RANDOM_START_BALL_SPEED
        b.velocity = pygame.math.Vector2(*rng.uniform(-v, v, size=2))
        self.prev_ball_pos = np.array(b.rect.center, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.set_state(self._initial_state)
        if self.random_start:
            self._randomize_start()
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        return np.array(
            [
                self.character.rect.centerx,
                self.character.rect.centery,
                self.character.dx,
                self.character.dy,
                self.ball.rect.centerx,
                self.ball.rect.centery,
                self.ball.velocity[0],
                self.ball.velocity[1],
                self.character.jumping,
                self.character.sprinting,
            ],
            dtype=np.float32,
        )

    def _simulate(self, action):
        """Advance the game one SIM_DT tick (no reward, no rendering)."""
        # 0: No-op
        if action == 1:  # Left
            self.character.key_down(pygame.K_LEFT)
        elif action == 2:  # Right
            self.character.key_down(pygame.K_RIGHT)
        elif action == 3:  # Jump
            self.character.key_down(pygame.K_UP)
        elif action == 4:  # Sprint
            self.character.key_down(pygame.K_LSHIFT)

        # Update game physics
        self.character.update(self.SIM_DT)
        self.ball.update(self.SIM_DT)

        character_hits_ball(self.character, self.ball)

    def step(self, action):
        self._simulate(action)

        # Reward
        reward = 0
        done = False

        ball_pos = np.array([self.ball.rect.centerx, self.ball.rect.centery])
        tgt_pos = np.array([self.target.rect.centerx, self.target.rect.centery])

        if ball_hits_target(self.ball, self.target):  # win condition
            reward += 10.0
            done = True
        else:
            dist = np.linalg.norm(ball_pos - tgt_pos)

            # penalty for each step
            reward -= 0.01

            # progress
            # prev_dist = np.linalg.norm(self.prev_ball_pos - tgt_pos)
            # if dist < prev_dist:
            #     reward += 0.1

            # distance to target
            # if dist < 100:
            #     reward -= 0.001 * dist

            # hit the ball
            # if self.character.can_hit_ball(self.ball):
            #     reward += 0.2

            # penalize special character movements
            # if self.character.jumping:
            #     reward -= 0.1

            # penalize special character movements
            # if self.character.sprinting:
            #     reward -= 0.1

        self.prev_ball_pos = ball_pos

        obs = self._get_observation()
        # SB3 logs rollout/success_rate from this key at episode end
        # (False on timeouts too, so the rate is unbiased).
        info = {"is_success": done}

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info

    def render(self):
        self.screen.fill(COLORS["sky_blue"])
        self.level.draw(self.screen)
        for ent in (self.character, self.ball, self.target):
            ent.draw(self.screen)
        pygame.event.pump()  # keep the window responsive (OS event queue)
        pygame.display.flip()
        # Throttle to real time only for viewing; physics use SIM_DT, not this.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()
