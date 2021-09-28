import cv2
import imutils
import numpy as np
import pygame
import pygame.freetype

from main import DISPLAY_SIZE

pygame.font.init()
pygame.mixer.init()

WIDTH, HEIGHT = DISPLAY_SIZE
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('EMOTION RECOGNITION')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 60

SITUATION_WIDTH = int(WIDTH * 0.3)
SITUATION_HEIGHT = int(HEIGHT * 0.4)

EMOTION_WIDTH = int(WIDTH * 0.1)
EMOTION_HEIGHT = int(HEIGHT * 0.15)

DIST = int(WIDTH * 0.06)

scheme_emoji = 'images/{}_emoji.png'

situation1 = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.imread('images/situation1.jpeg')[:, :, ::-1], width=SITUATION_WIDTH), k=1))
situation2 = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.imread('images/situation2.png')[:, :, ::-1], width=SITUATION_WIDTH), k=1))
situation3 = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.imread('images/situation3.jpeg')[:, :, ::-1], width=SITUATION_WIDTH), k=1))
situation4 = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.imread('images/situation4.jpeg')[:, :, ::-1], width=SITUATION_WIDTH), k=1))

# emotions
happy = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('happy'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))
sad = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('sad'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))
angry = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('angry'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))
scared = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('scared'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))
disgusted = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('disgusted'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))
surprised = pygame.surfarray.make_surface(np.rot90(imutils.resize(cv2.flip(cv2.imread(scheme_emoji.format('surprised'))[:, :, ::-1], 1), width=EMOTION_WIDTH), k=1))

X_START = WIDTH * 0.35
Y_START = HEIGHT * 0.1

myfont = pygame.font.SysFont('Comic Sans MS', 60)


def draw_window(situation, situation1, happy_, sad_, angry_, scared_, disgusted_, surprised_):
    WIN.fill(WHITE)
    WIN.blit(myfont.render('Choose emotion', False, (0, 0, 0)), (X_START * 1.2, 25))
    WIN.blit(situation1, (situation.x, situation.y))
    WIN.blit(happy, (happy_.x, happy_.y))
    WIN.blit(sad, (sad_.x, sad_.y))
    WIN.blit(angry, (angry_.x, angry_.y))
    WIN.blit(scared, (scared_.x, scared_.y))
    WIN.blit(disgusted, (disgusted_.x, disgusted_.y))
    WIN.blit(surprised, (surprised_.x, surprised_.y))
    pygame.display.update()


def start_game():
    situation = pygame.Rect(X_START, Y_START, SITUATION_WIDTH, SITUATION_HEIGHT)
    happy_ = pygame.Rect(DIST, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)
    sad_ = pygame.Rect(DIST * 2 + EMOTION_WIDTH, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)
    angry_ = pygame.Rect(DIST * 3 + EMOTION_WIDTH * 2, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)
    scared_ = pygame.Rect(DIST * 4 + EMOTION_WIDTH * 3, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)
    disgusted_ = pygame.Rect(DIST * 5 + EMOTION_WIDTH * 4, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)
    surprised_ = pygame.Rect(DIST * 6 + EMOTION_WIDTH * 5, Y_START + SITUATION_HEIGHT + EMOTION_HEIGHT, EMOTION_WIDTH, EMOTION_HEIGHT)

    pictures = [(situation1, scared_), (situation2, sad_), (situation3, disgusted_), (situation4, surprised_)]

    clock = pygame.time.Clock()

    run = True
    click = 0
    current = 0
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if pictures[current][1].x + pictures[current][1].width >= x >= pictures[current][1].x and \
                        pictures[current][1].y + pictures[current][1].height >= y >= pictures[current][1].y:
                    click = 1
                    # current += 1
                else:
                    click = -1
                    # click = 0

        if current >= len(pictures):
            WIN.blit(myfont.render('Good job!!!', False, (0, 0, 0)), (X_START + SITUATION_WIDTH + 50, HEIGHT // 2))
            pygame.display.update()
            pygame.time.wait(2000)
            run = False
        else:
            draw_window(situation, pictures[current][0], happy_, sad_, angry_, scared_, disgusted_, surprised_)
            if click == 1:
                WIN.blit(myfont.render('Right!', False, (0, 0, 0)), (X_START + SITUATION_WIDTH + 50, HEIGHT // 2))
                pygame.display.update()
                pygame.time.wait(1000)
                WIN.blit(myfont.render('Right!', False, (255, 255, 255)), (X_START + SITUATION_WIDTH + 50, HEIGHT // 2))
                pygame.display.update()
                current += click
                click = 0
            elif click == -1:
                WIN.blit(myfont.render('Try another emotion!', False, (0, 0, 0)), (X_START + SITUATION_WIDTH + 50, HEIGHT // 2))
                pygame.display.update()
                pygame.time.wait(1000)
                WIN.blit(myfont.render('Try another emotion!', False, (255, 255, 255)), (X_START + SITUATION_WIDTH + 50, 600))
                pygame.display.update()
                click = 0
                current += click

    return 0


if __name__ == '__main__':
    start_game()
