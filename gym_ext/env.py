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

    def __init__(self, render_mode=None):
        super().__init__()
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.character.reset()
        self.ball.reset()
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

    def step(self, action):
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
        info = {}

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
