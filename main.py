import cv2
import imutils
import numpy as np

SCALE_MIN = 0.2
SCALE_MAX = 1.2
SCALE_STEP_DIFF = 0.02
SCALE_NUM = int((SCALE_MAX - SCALE_MIN) / SCALE_STEP_DIFF)
ROTATE_STEP_DIFF = 15
THRESHOLD = 0.8
RECTANGLE_THICKNESS = 2


def finding(source_image_path, template_list, bgr_color_list):
    img_rgb = cv2.imread(source_image_path)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)

    for index, template in enumerate(template_list):
        img_rgb = process(template, img_rgb, img_gray, bgr_color_list[index])

    cv2.imwrite("template_matched.jpg", img_rgb)
    cv2.imshow("Matched image", img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def process(template_image_path, img_rgb, img_gray, color):
    template = cv2.imread(template_image_path, 0)
    template = cv2.bilateralFilter(template, 11, 17, 17)

    for scale in np.linspace(SCALE_MIN, SCALE_MAX, SCALE_NUM)[::-1]:
        resized = imutils.resize(template, width=int(template.shape[1] * scale))

        for angle in np.arange(0, 360, ROTATE_STEP_DIFF):
            rotated_img = imutils.rotate(resized, angle)
            w, h = rotated_img.shape[::-1]
            res = cv2.matchTemplate(img_gray, rotated_img, cv2.TM_CCOEFF_NORMED)

            threshold = THRESHOLD
            loc = np.where(res >= threshold)

            if loc:
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img_rgb, pt, (pt[0] + w + 4, pt[1] + h + 4), color=color,
                                  thickness=RECTANGLE_THICKNESS)
    return img_rgb


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    list_template = [
        "template/xx.png", "template/bong.png", "template/buom.png",
        "template/dautay.png", "template/kem.png", "template/may.png",
        "template/oc_que.png", "template/pho_mai.png", "template/xuong.png",
        "template/chim.png", "template/no.png", "template/bo.png"
    ]
    bgr_color_list = [
        (0, 0, 0), (0, 0, 255), (255, 0, 0),
        (0, 204, 0), (255, 0, 255), (204, 51, 51),
        (204, 204, 51), (255, 125, 255), (0, 155, 0),
        (0, 204, 255), (102, 0, 102), (0, 50, 100)
    ]
    finding("finding_data/source.png", list_template, bgr_color_list)
