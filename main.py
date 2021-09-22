import time

import cv2
import pygame
import pygame.camera
import pygame.camera
from keras.models import load_model

from app import start_game
from face_alignment import FaceAligner

detection_model_path = '../Emotion-recognition/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '../Emotion-recognition/models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

face_aligner = FaceAligner()

EMOTIONS = ["ANGRY", "DISGUST", "SCARED", "HAPPY", "SAD", "SURPRISED", "NEUTRAL"]

light_blue = (9, 220, 205)
light_green = (170, 227, 136)
violet = (119, 74, 141)
yellow = (231, 230, 46)

res_coef = 1.2
camera_size = (int(1280 * res_coef), int(960 * res_coef))
BORDER_WIDTH = int(100 * res_coef)

DISPLAY_SIZE = (camera_size[0] * 2 + BORDER_WIDTH * 5, camera_size[1] + BORDER_WIDTH * 3)

START_POS = (DISPLAY_SIZE[0] // 2 - BORDER_WIDTH, DISPLAY_SIZE[1] // 2 - BORDER_WIDTH * 2)
EXIT_POS = (DISPLAY_SIZE[0] // 2 - BORDER_WIDTH, DISPLAY_SIZE[1] // 2 + BORDER_WIDTH)

pygame.init()
display = pygame.display.set_mode(DISPLAY_SIZE)


def draw_start(display):
    pygame.draw.rect(display, yellow,
                     (START_POS[0], START_POS[1], BORDER_WIDTH * 3, int(BORDER_WIDTH * 1.4)),
                     int(20 * res_coef), border_radius=int(30 * res_coef))

    pygame.draw.rect(display, light_green,
                     pygame.Rect(START_POS[0] + 10 * res_coef,
                                 START_POS[1] + 10 * res_coef,
                                 BORDER_WIDTH * 3 - 20 * res_coef,
                                 BORDER_WIDTH * 1.4 - 20 * res_coef),
                     int(60 * res_coef), border_radius=int(20 * res_coef))

    font = pygame.font.SysFont('Comic Sans MS', int(120 * res_coef))
    textsurface = font.render('START', False, (0, 0, 0))
    display.blit(textsurface, (START_POS[0] + 20 * res_coef, START_POS[1] + 33 * res_coef))


def draw_exit(display):
    pygame.draw.rect(display, yellow,
                     (EXIT_POS[0], EXIT_POS[1], BORDER_WIDTH * 3, int(BORDER_WIDTH * 1.4)),
                     int(20 * res_coef), border_radius=int(30 * res_coef))

    pygame.draw.rect(display, light_green,
                     pygame.Rect(EXIT_POS[0] + 10 * res_coef,
                                 EXIT_POS[1] + 10 * res_coef,
                                 BORDER_WIDTH * 3 - 20 * res_coef,
                                 BORDER_WIDTH * 1.4 - 20 * res_coef),
                     int(60 * res_coef), border_radius=int(20 * res_coef))

    font = pygame.font.SysFont('Comic Sans MS', int(120 * res_coef))
    textsurface = font.render('EXIT', False, (0, 0, 0))
    display.blit(textsurface, (EXIT_POS[0] + 52 * res_coef, EXIT_POS[1] + 33 * res_coef))


def start_screen(display):
    display.fill(violet)
    draw_start(display)
    draw_exit(display)
    pygame.display.update()


if __name__ == '__main__':
    while True:
        start_screen(display)
        pressed = False
        code = 0
        while not pressed:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if START_POS[0] < pos[0] < START_POS[0] + BORDER_WIDTH * 3 and \
                       START_POS[1] < pos[1] < START_POS[1] + BORDER_WIDTH * 1.4:
                        code = start_game()
                        pressed = True
                        break

                    elif EXIT_POS[0] < pos[0] < EXIT_POS[0] + BORDER_WIDTH * 3 and \
                         EXIT_POS[1] < pos[1] < EXIT_POS[1] + BORDER_WIDTH * 1.4:
                        exit()

        if code == 1:
            break

        time.sleep(2)
