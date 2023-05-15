import multiprocessing
from rehab_games.home_screen import game
from hand_gesture_recognition import run_HGR
import pygame


if __name__ == "__main__":
  # start the gesture recognition process
  p = multiprocessing.Process(target=run_HGR)
  p.start()

  screen = pygame.display.set_mode((1280, 720))
  game(screen)
