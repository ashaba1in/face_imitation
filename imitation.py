import os
import time

import cv2
import face_recognition
import imutils
import numpy as np
import pygame
import pygame.camera
import pygame.camera
import pygame.camera
from pygame.locals import *

from face_alignment import FaceAligner
from main import res_coef

face_aligner = FaceAligner()

EMOTIONS = ["ANGRY", "DISGUST", "SCARED", "HAPPY", "SAD", "SURPRISED", "NEUTRAL"]

red = (255, 0, 0)
green = (0, 255, 0)
black = (0, 0, 0)
light_blue = (9, 220, 205)
light_green = (170, 227, 136)
violet = (119, 74, 141)
yellow = (231, 230, 46)

camera_size = (int(1280 * res_coef), int(960 * res_coef))
BORDER_WIDTH = int(100 * res_coef)

FACE_POS = (camera_size[0] + BORDER_WIDTH * 4, BORDER_WIDTH)
REF_FACE_POS = (BORDER_WIDTH, BORDER_WIDTH)
SCALE_POS = (BORDER_WIDTH * 2 + camera_size[0], BORDER_WIDTH)
SKIP_POS = (BORDER_WIDTH * 9 + camera_size[0], BORDER_WIDTH + camera_size[1] + 30 * res_coef)

STAR_POS = (BORDER_WIDTH, BORDER_WIDTH + camera_size[1] + 30 * res_coef)
STAR_POINTS = np.array([[165, 151], [200, 20], [235, 151], [371, 144], [257, 219],
                        [306, 346], [200, 260], [94, 346], [143, 219], [29, 144]]) * 0.4 * res_coef
NUM_STARS = 5

pygame.init()
pygame.camera.init()

# 300 - essential for propper work DO NOT TOUCH!!!!!!!
# just an input image width for NN to work faster
face_coef = camera_size[0] / 300.

cam = pygame.camera.Camera("/dev/video0", camera_size)
cam.start()
display = pygame.display.set_mode((camera_size[0] * 2 + BORDER_WIDTH * 5, camera_size[1] + BORDER_WIDTH * 3))

SUCCESS_THRESHOLD = 0.001
MAX_SCORE = 0.1


def draw_text(display, text):
    font = pygame.font.SysFont('Comic Sans MS', int(120 * res_coef))
    textsurface = font.render(text.upper(), False, red)
    display.blit(textsurface, (REF_FACE_POS[0] + int(52 * res_coef), REF_FACE_POS[1] + int(33 * res_coef)))


def draw_skip(display):
    pygame.draw.rect(display, yellow,
                     (SKIP_POS[0], SKIP_POS[1], BORDER_WIDTH * 3, int(BORDER_WIDTH * 1.4)),
                     int(20 * res_coef), border_radius=int(30 * res_coef))

    pygame.draw.rect(display, light_green,
                     pygame.Rect(SKIP_POS[0] + 10 * res_coef,
                                 SKIP_POS[1] + 10 * res_coef,
                                 BORDER_WIDTH * 3 - 20 * res_coef,
                                 BORDER_WIDTH * 1.4 - 20 * res_coef),
                     int(60 * res_coef), border_radius=int(20 * res_coef))

    font = pygame.font.SysFont('Comic Sans MS', int(120 * res_coef))
    textsurface = font.render('SKIP', False, (0, 0, 0))
    display.blit(textsurface, (SKIP_POS[0] + int(52 * res_coef), SKIP_POS[1] + int(33 * res_coef)))


def draw_scale(display, score):
    x, y = SCALE_POS

    # remove all
    pygame.draw.rect(display, violet, pygame.Rect(x, y, BORDER_WIDTH, camera_size[1]))

    ratio = score / MAX_SCORE

    # draw scale
    scale_height = ratio * camera_size[1]
    pygame.draw.rect(display, light_blue, pygame.Rect(x, y + scale_height, BORDER_WIDTH, camera_size[1] - scale_height))

    # draw threshold
    win_threshold = SUCCESS_THRESHOLD / MAX_SCORE * camera_size[1]
    pygame.draw.line(display, red, [x, y + win_threshold], [x + BORDER_WIDTH, y + win_threshold], 4)

    # draw borders
    pygame.draw.rect(display, black, (x, y, BORDER_WIDTH, camera_size[1]), int(10 * res_coef))


def get_similarity_score(x1, x2):
    return 1 - np.sum(x1 * x2) / np.sqrt(np.sum(x1 ** 2)) / np.sqrt(np.sum(x2 ** 2))


def get_norm_landmarks(img):
    landmarks = face_recognition.face_landmarks(img, model='large')
    if len(landmarks) == 0:
        return None

    M = face_aligner.align(
        img,
        np.array(landmarks[0]['left_eye']),
        np.array(landmarks[0]['right_eye'])
    )

    del landmarks[0]['nose_bridge']
    del landmarks[0]['nose_tip']
    del landmarks[0]['left_eye']
    del landmarks[0]['right_eye']
    del landmarks[0]['chin']

    all_marks = np.array([mark for marks in landmarks[0].values() for mark in marks]).astype(float)

    all_marks = (M[:, :2] @ all_marks.T).T + M[:, 2]

    min_ = np.min(np.array(all_marks), 0)
    all_marks -= min_

    w, h = all_marks.max(0) - all_marks.min(0)
    all_marks = all_marks / np.array([w, h])

    return all_marks


def get_landmarks_score(ref_marks, marks):
    if ref_marks is None or marks is None:
        return MAX_SCORE  # more then threshold

    score = np.mean((ref_marks - marks) ** 2)

    return min(score, MAX_SCORE)


def get_preds(img):
    img = imutils.resize(img, width=300)

    landmarks = get_norm_landmarks(img)

    return landmarks


def get_random_img_path():
    paths = list(os.walk('images'))[1:]
    person_path = paths[np.random.randint(len(paths))]
    new_img_path = os.path.join(person_path[0], np.random.choice(person_path[2]))

    return new_img_path


def get_ref_image_data(display):
    img_path = get_random_img_path()
    ref_img = cv2.imread(img_path)[:, :, ::-1]

    landmarks = get_preds(ref_img)

    ref_img = cv2.resize(ref_img, camera_size)

    ref_surf = pygame.surfarray.make_surface(np.rot90(ref_img, k=1))

    display.blit(ref_surf, REF_FACE_POS)
    draw_text(display, img_path.split('/')[-1].split('.')[0])

    return ref_surf, landmarks


def start_game():
    success_times = 0

    display.fill(violet)
    draw_skip(display)

    ref_surf, ref_landmarks = get_ref_image_data(display)

    for i in range(NUM_STARS):
        star_pos = STAR_POINTS + STAR_POS
        star_pos[:, 0] += i * BORDER_WIDTH * 2
        pygame.draw.polygon(display, yellow, star_pos, 4)

    MEANS_SIZE = 10
    scores = [MAX_SCORE] * MEANS_SIZE

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                return 1

            if e.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if SKIP_POS[0] < pos[0] < SKIP_POS[0] + BORDER_WIDTH * 3 and \
                        SKIP_POS[1] < pos[1] < SKIP_POS[1] + BORDER_WIDTH * 1.4:
                    ref_surf, ref_landmarks = get_ref_image_data(display)
                    scores = [MAX_SCORE] * MEANS_SIZE

        snapshot = cam.get_image()
        snapshot = pygame.transform.scale(snapshot, camera_size)
        face_array = np.rot90(pygame.surfarray.array3d(snapshot), k=3)

        # show face
        display.blit(snapshot, FACE_POS)

        norm_landmarks = get_preds(face_array)

        # Code for drawing landmarks of face

        landmarks = face_recognition.face_landmarks(imutils.resize(face_array, width=300))
        if landmarks:
            # del landmarks[0]['nose_bridge']
            # del landmarks[0]['nose_tip']
            # del landmarks[0]['left_eye']
            # del landmarks[0]['right_eye']
            # del landmarks[0]['chin']
            for marks in landmarks[0].values():
                for mark in marks:
                    mark = np.array(mark)
                    mark = mark * face_coef
                    mark[0] = camera_size[0] - mark[0]
                    mark += FACE_POS
                    pygame.draw.circle(display, green, mark, 3, width=2)

        score = get_landmarks_score(ref_landmarks, norm_landmarks)

        scores.append(score)
        scores = scores[-MEANS_SIZE:]

        draw_scale(display, np.mean(scores))

        pygame.display.update()

        if np.mean(scores) < SUCCESS_THRESHOLD:
            star_pos = STAR_POINTS + STAR_POS
            star_pos[:, 0] += success_times * BORDER_WIDTH * 2
            pygame.draw.polygon(display, yellow, star_pos)
            success_times += 1

            pygame.display.update()

            i = np.random.randint(1, 5)
            pygame.mixer.music.load(f'sounds/good_job{i}.wav')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pass

            if success_times == NUM_STARS:
                return 0

            ref_surf, ref_landmarks = get_ref_image_data(display)

            scores = [MAX_SCORE] * MEANS_SIZE

            time.sleep(1)
