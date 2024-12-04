import math
import pygame
import random
import numpy as np
from nn import DDPG

WIDTH, HEIGHT = 1920, 1080

DRONES = 1
TARGETS = 20

FPS = 60

IMAGE_DRONE_W = 240
IMAGE_DRONE_H = 120
IMAGE_ENGINE_W = 140
IMAGE_ENGINE_H = 120
X_OFFSET = 80
Y_OFFSET = 24

NUM_ENGINE_IMAGES = 16
NUM_DRONE_IMAGES = 16

BASE_X, BASE_Y = 960, 900
DISTANCE = 15
GRAVITY = 0.05
MAX_FALL_SPEED = 5
DOWN_LIMIT = 50

TARGETS_DISTANCE = 200

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


class AssetManager:
    def __init__(self):
        self.drone_images = self.load_images(
            'drone_body', IMAGE_DRONE_W, IMAGE_DRONE_H, NUM_DRONE_IMAGES)
        self.engine_images = self.load_images(
            'engine', IMAGE_ENGINE_W, IMAGE_ENGINE_H, NUM_ENGINE_IMAGES)

    def load_images(self, prefix, width, height, count):
        return [pygame.transform.smoothscale(pygame.image.load(f'images/{prefix}{i}.png').convert_alpha(), (width, height)) for i in range(1, count + 1)]


class Engine:
    def __init__(self, position, index=0, assets=None):
        self.position = position
        self.angle = 0
        self.index = index
        self.size = (IMAGE_ENGINE_W, IMAGE_ENGINE_H)
        self.images = assets.engine_images
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.x, self.y = self.position
        self.rect.topleft = self.position
        self.rotation_cache = {}

    def update(self, screen):
        self.image = self.images[self.index]
        self.index = (self.index + 1) % NUM_ENGINE_IMAGES
        self.rotate_center(screen)

    def rotate_center(self, screen):
        angle_deg = int(math.degrees(self.angle)) % 360
        # Create a unique cache key using both the angle and the index
        cache_key = (angle_deg, self.index)

        # Check if the rotated image is already in the cache
        if cache_key not in self.rotation_cache:
            rotated_image = pygame.transform.rotate(self.image, -angle_deg)
            self.rotation_cache[cache_key] = rotated_image
        else:
            rotated_image = self.rotation_cache[cache_key]

        # Update rect position with rotated image
        self.rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, self.rect.topleft)

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.rect.center = (self.x, self.y)


class Drone(pygame.sprite.Sprite):
    def __init__(self, delta_time=1, agent=None, assets=None):
        super().__init__()
        self.assets = assets
        self.delta_time = delta_time
        self.speed = 0.5 * delta_time
        self.vertical_speed = 0
        self.agent = agent
        self.index = 0
        self.size = (IMAGE_DRONE_W, IMAGE_DRONE_H)
        self.images = self.assets.drone_images
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.x, self.y = BASE_X, BASE_Y
        self.angle = 0
        self.target = None
        self.nearest_target = None
        self.engine_left = None
        self.engine_right = None
        self.flag = 0
        self.steps = 0
        self.prev_state = None
        self.engine_left_diff = 0
        self.engine_right_diff = 0
        self.create_engines()

    def create_engines(self):
        self.engine_left = Engine(
            position=(BASE_X - X_OFFSET, BASE_Y - Y_OFFSET), assets=self.assets)
        self.engine_right = Engine(
            position=(BASE_X + X_OFFSET, BASE_Y - Y_OFFSET), index=8, assets=self.assets)

    def get_drone_coordinates(self):
        self.x = (self.engine_left.x + self.engine_right.x) / 2
        self.y = (self.engine_left.y + self.engine_right.y) / 2 + Y_OFFSET
        self.rect.center = (self.x, self.y)

    def reset(self):
        self.create_engines()
        self.get_drone_coordinates()

    def update(self, screen, targets):
        self.get_drone_coordinates()

        state = self.get_state()
        angle_change = self.agent.get_action(state)
        if isinstance(angle_change, np.ndarray) and angle_change.size == 1:
            angle_change = angle_change.item()

        self.hand_movement(angle_change)
        self.draw(screen)
        self.update_engine(screen)
        if self.is_collision():
            self.reset()
        self.highlight_nearest_target(screen, targets)

        if self.check_collision(targets):
            self.done = 1
        else:
            self.done = 0
        next_state = self.get_state()
        reward = self.get_reward()

        if self.agent and self.prev_state:
            self.agent.fit(self.prev_state, angle_change,
                           reward, self.done, next_state)
        self.prev_state = next_state

        # self.steps += 1
        # if self.steps % 1000 == 0:
        #     self.agent.save_model(
        #         filepath="ddpg_checkpoint.pth")

    def update_engine(self, screen):
        self.engine_left.update(screen)
        self.engine_right.update(screen)

    def highlight_nearest_target(self, screen, targets):
        if targets:
            self.nearest_target = min(
                targets, key=lambda t: math.hypot(t.x - self.x, t.y - self.y))
            for target in targets:
                if target == self.nearest_target:
                    target.draw(screen, RED)
                else:
                    target.draw(screen, GREEN)
        else:
            self.nearest_target = None

    def hand_movement(self, angle_change):
        self.image = self.images[self.index]
        self.index = (self.index + 1) % NUM_DRONE_IMAGES

        thrust_applied = False
        if self.nearest_target:
            if self.flag == 0:
                self.apply_thrust()
                thrust_applied = True
                if self.nearest_target.y >= self.engine_left.y + Y_OFFSET:
                    self.flag = 1
            elif self.flag == 1:
                if self.engine_left.y < self.nearest_target.y + DOWN_LIMIT:
                    self.apply_gravity()
                    self.smoothly_vertical(0.025)
                else:
                    self.flag = 0
            self.adjust_engines_to_target(angle_change)
        else:
            if abs(self.x - WIDTH // 2) > 10 and abs(self.y - HEIGHT // 2) > 10:
                self.apply_thrust()
                self.adjust_engines_to_target(angle_change)
            else:
                self.smoothly_vertical(0.025)
                self.apply_gravity()
        if not thrust_applied and self.y >= BASE_Y:
            self.vertical_speed = 0

    def apply_thrust(self):
        slow_factor = 1
        if self.nearest_target:
            distance_to_target = self.distance_to_target()
            slow_factor = min(slow_factor, (distance_to_target / 200) ** 0.5)

        left_dx, left_dy = self.engine_movement(
            self.engine_left.angle, slow_factor)
        right_dx, right_dy = self.engine_movement(
            self.engine_right.angle, slow_factor)
        self.engine_left.set_position(
            self.engine_left.x + left_dx, self.engine_left.y + left_dy - GRAVITY)
        self.engine_right.set_position(
            self.engine_right.x + right_dx, self.engine_right.y + right_dy - GRAVITY)

    def apply_gravity(self):
        if self.y <= BASE_Y:
            self.vertical_speed += GRAVITY
            self.vertical_speed = min(self.vertical_speed, MAX_FALL_SPEED)
            self.engine_left.set_position(
                self.engine_left.x, self.engine_left.y + self.vertical_speed)
            self.engine_right.set_position(
                self.engine_right.x, self.engine_right.y + self.vertical_speed)

    def engine_movement(self, angle, slow_factor):
        return self.speed * slow_factor * math.sin(angle), -self.speed * slow_factor * math.cos(angle)

    def adjust_engines_to_target(self, angle_change):
        if self.nearest_target:
            target_pos = np.array(
                (self.nearest_target.x, self.nearest_target.y))
        else:
            target_pos = np.array((WIDTH // 2, HEIGHT // 2))

        angle_to_target = math.atan2(
            target_pos[1] - self.y, target_pos[0] - self.x)

        self.engine_left_diff = (angle_to_target - self.engine_left.angle -
                                 0.5 * math.pi) % (2 * math.pi) - math.pi
        self.engine_right_diff = (angle_to_target - self.engine_right.angle -
                                  0.5 * math.pi) % (2 * math.pi) - math.pi

        self.engine_left.angle += angle_change * 0.05
        self.engine_left.angle = self.engine_left.angle % (2 * math.pi)
        self.engine_right.angle += angle_change * 0.05
        self.engine_right.angle = self.engine_right.angle % (2 * math.pi)

    def smoothly_vertical(self, offset):
        if self.engine_left.angle > 0 or self.engine_right.angle > 0:
            self.engine_left.angle = max(0, self.engine_left.angle - offset)
            self.engine_right.angle = max(0, self.engine_right.angle - offset)
        elif self.engine_left.angle < 0 or self.engine_right.angle < 0:
            self.engine_left.angle = min(0, self.engine_left.angle + offset)
            self.engine_right.angle = min(0, self.engine_right.angle + offset)

    def is_collision(self):
        return (self.rect.left < 0 or self.rect.right > WIDTH or
                self.rect.top < 0 or self.rect.bottom > HEIGHT)

    def check_collision(self, targets):
        for target in targets:
            if target.rect.collidepoint(self.x, self.y):
                target.kill()
                target = None
                return True
        return False

    def draw(self, screen):
        self.rect.center = (self.x, self.y)
        screen.blit(self.image, self.rect)

    def angle_to_target(self):
        angle_to_target = math.atan2(
            self.nearest_target.y - self.y,
            self.nearest_target.x - self.x) if self.nearest_target else 0
        return angle_to_target

    def distance_to_target(self):
        distance_to_target = math.hypot(
            self.nearest_target.x - self.x, self.nearest_target.y - self.y)
        return distance_to_target

    def get_state(self):
        self.clamp_position()

        # Angle to nearest target
        angle_to_target = self.angle_to_target()

        # Normalization of drone coordinates
        normalized_x = self.x / WIDTH
        normalized_y = self.y / HEIGHT

        # Normalize the angle to the target
        normalized_angle_to_target = (
            self.normalize_angle(angle_to_target) + np.pi) / (2 * np.pi)

        # State
        state = [
            normalized_x, normalized_y,
            normalized_angle_to_target,
            self.engine_left_diff
        ]
        return state

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def clamp_position(self):
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def get_reward(self):
        reward = 0
        max_dist = np.linalg.norm([WIDTH, HEIGHT])

        # Distance to the nearest target
        if self.nearest_target:
            distance_to_target = self.distance_to_target()
        else:
            distance_to_target = WIDTH  # if there are no targets, the maximum distance

        # Reward for precise direction towards the goal
        reward += 5*math.cos(self.engine_right_diff) + 1
        reward += 6*(1 - min(distance_to_target / max_dist, 1))

        # Reward for no collisions
        if not self.is_collision():
            reward += 15
        else:
            reward -= 10  

        # Reward for achieving the goal
        if self.done:
            reward += 100

        return reward


class Targets(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.target = pygame.Surface((30, 30))
        self.target.set_colorkey((0, 0, 0))
        self.rect = self.target.get_rect()
        self.target_mask = pygame.mask.from_surface(self.target)

    def draw(self, screen, color=GREEN):
        self.rect.center = (self.x, self.y)
        self.target.fill(BLACK)
        pygame.draw.circle(self.target, color, (12, 12), 10)
        screen.blit(self.target, self.rect)


class Setup:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('AI Drone')
        icon = pygame.image.load('images/drone_icon.png')
        pygame.display.set_icon(icon)
        self.clock = pygame.time.Clock()
        self.delta_time = self.clock.tick(FPS)
        self.assets = AssetManager()

        self.agent = None

        self.reset_game()

    def reset_game(self):
        self.agent = DDPG(state_dim=4, action_dim=1,
                          action_scale=1, noise_decrease=0.00005)
        self.drones = pygame.sprite.Group(
            [Drone(self.delta_time, self.agent, assets=self.assets) for _ in range(DRONES)])
        self.targets = pygame.sprite.Group(
            [Targets(random.uniform(TARGETS_DISTANCE, WIDTH-TARGETS_DISTANCE),
                     random.uniform(TARGETS_DISTANCE, HEIGHT-TARGETS_DISTANCE-300)) for _ in range(TARGETS)])
        # self.agent.load_model(filepath="ddpg_checkpoint.pth")

    def update_objects(self):
        for target in self.targets:
            target.draw(self.screen)
        for drone in self.drones:
            drone.update(self.screen, self.targets)

    def main(self):
        while True:
            self.screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset_game()

            self.update_objects()

            pygame.display.update()
            self.clock.tick(FPS)


if __name__ == '__main__':
    setup_instance = Setup()
    setup_instance.main()
