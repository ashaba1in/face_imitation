import time

import pygame
import pygame.camera
import pygame.camera

light_green = (170, 227, 136)
violet = (119, 74, 141)
yellow = (231, 230, 46)

res_coef = 0.6  # coefficient for display resolution
camera_size = (int(1280 * res_coef), int(960 * res_coef))
BORDER_WIDTH = int(100 * res_coef)

DISPLAY_SIZE = (camera_size[0] * 2 + BORDER_WIDTH * 5, camera_size[1] + BORDER_WIDTH * 3)

IMITATION_POS = (DISPLAY_SIZE[0] // 2 - BORDER_WIDTH * 7, DISPLAY_SIZE[1] // 2 - BORDER_WIDTH * 2)
RECOGNITION_POS = (DISPLAY_SIZE[0] // 2 + BORDER_WIDTH * 3, DISPLAY_SIZE[1] // 2 - BORDER_WIDTH * 2)
EXIT_POS = (DISPLAY_SIZE[0] // 2 - BORDER_WIDTH, DISPLAY_SIZE[1] // 2 + BORDER_WIDTH)

pygame.init()
display = pygame.display.set_mode(DISPLAY_SIZE)


# this import is here to solve circular import problem
import imitation
import recognition


def draw_text(display, text, pos):
    pygame.draw.rect(display, yellow,
                     (pos[0], pos[1], BORDER_WIDTH * 6, int(BORDER_WIDTH * 1.4)),
                     int(20 * res_coef), border_radius=int(30 * res_coef))

    pygame.draw.rect(display, light_green,
                     pygame.Rect(pos[0] + 10 * res_coef,
                                 pos[1] + 10 * res_coef,
                                 BORDER_WIDTH * 6 - 20 * res_coef,
                                 BORDER_WIDTH * 1.4 - 20 * res_coef),
                     int(60 * res_coef), border_radius=int(20 * res_coef))

    font = pygame.font.SysFont('Comic Sans MS', int(120 * res_coef))
    textsurface = font.render(text, False, (0, 0, 0))
    display.blit(textsurface, (pos[0] + 20 * res_coef, pos[1] + 33 * res_coef))


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
    draw_text(display, 'RECOGNITION', RECOGNITION_POS)
    draw_text(display, 'IMITATION', IMITATION_POS)
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
                    if IMITATION_POS[0] < pos[0] < IMITATION_POS[0] + BORDER_WIDTH * 6 and \
                       IMITATION_POS[1] < pos[1] < IMITATION_POS[1] + BORDER_WIDTH * 1.4:
                        code = imitation.start_game()
                        pressed = True
                        break
                    elif RECOGNITION_POS[0] < pos[0] < RECOGNITION_POS[0] + BORDER_WIDTH * 6 and \
                         RECOGNITION_POS[1] < pos[1] < RECOGNITION_POS[1] + BORDER_WIDTH * 1.4:
                        code = recognition.start_game()
                        pressed = True
                        break
                    elif EXIT_POS[0] < pos[0] < EXIT_POS[0] + BORDER_WIDTH * 3 and \
                         EXIT_POS[1] < pos[1] < EXIT_POS[1] + BORDER_WIDTH * 1.4:
                        exit()

        time.sleep(1)
