import multiprocessing
from rehab_games import Game
from hand_gesture_recognition import run_HGR
import pygame

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

if __name__ == "__main__":
  # start the gesture recognition process
  p = multiprocessing.Process(target=run_HGR)
  p.start()

  # init pygame parameters
  pygame.init()
  pygame.font.init()
  pygame.font.get_init()
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  clock = pygame.time.Clock()

  # start the game
  game = Game(screen, SCREEN_WIDTH, SCREEN_HEIGHT, clock)
  game.play()

  p.join()