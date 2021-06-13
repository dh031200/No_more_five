import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import ImageGrab
import cv2
import numpy as np
from scipy.spatial import distance

# Set option
threshold = 0.7
input_size = 416
left_top_x = 200
left_top_y = 200
detect_width = 1280
detect_height = 720

# Set path
model_path = 'models/'

# Set file name
class_info = 'obj.names'
model = 'yolov4-0613'  # 사용할 모델 'yolov4-first', 'yolov4-0613'

# Variables
weights = model_path + model
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
Five = False


def find_nearest(tables, point):
    nearest_index = distance.cdist([point], tables).argmin()
    return nearest_index  # 인덱스 반환


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(s_image, bboxes, classes_name=None, show_label=True, five=False):
    if classes_name is None:
        classes_name = read_class_names(class_info)
    num_classes = len(classes_name)
    image_h, image_w, _ = s_image.shape
    colors = [[255, 128, 0], [128, 255, 128]]
    people_coords = []
    table_coords = []
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    classes_cnt = [0] * num_classes
    table_cnt = 1
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])

        # 클래스별로 카운팅을 위함
        print(classes_name[class_ind])
        classes_cnt[class_ind] += 1
        # print("left_top : ", c1, ", right_bottom: ", c2)

        # 박스 중앙 점 x, y 좌표 계산
        center_x = int((c1[0] + c2[0]) / 2)
        center_y = int((c1[1] + c2[1]) / 2)
        print("x: ", center_x, ", y: ", center_y)

        # 클래스별 좌표 저장
        if classes_name[class_ind] == "Person":
            people_coords.append([center_x, center_y])
        elif classes_name[class_ind] == "Table":
            table_coords.append([center_x, center_y])
        print()
        # boxing object
        cv2.rectangle(s_image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            if classes_name[class_ind] == 'Table':
                bbox_mess = '%s_%d: %.2f' % (classes_name[class_ind], table_cnt, score)
                table_cnt += 1
            else:
                bbox_mess = '%s: %.2f' % (classes_name[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(s_image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled

            cv2.putText(s_image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    # 각 클래스별로 감지된 객체 수 출력
    print(people_coords)
    print(classes_cnt[0], " people, ", classes_cnt[1], " tables")

    if classes_cnt[0] == 0 or classes_cnt[1] == 0:
        return s_image, five
    # group 판별
    group = [0] * classes_cnt[1]
    for i in people_coords:
        group[find_nearest(table_coords, i)] += 1

    # 콘솔 출력용
    print_group = ['그룹' + str(i + 1) for i in range(classes_cnt[1])]
    print(*print_group)
    print(' ', end='')
    print(*group, sep='명　 ', end='명')
    print('\n')

    # 화면에 테이블당 인원 출력
    for i in range(classes_cnt[1]):
        mess = 'Table_%d : %d' % (i + 1, group[i])
        if group[i] <= 3:
            t_color = (255, 255, 255)
        elif group[i] == 4:
            t_color = (30, 200, 200)
        else:
            t_color = (0, 0, 255)
        cv2.putText(s_image, mess, (40, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, t_color, 2, lineType=cv2.LINE_AA)
    if max(group) < 5:
        b_color = (255, 255, 255)
    else:
        b_color = (0, 0, 255)
    cv2.rectangle(s_image, (20, 20), (240, 30 + (classes_cnt[1] * 30)), b_color, 3)

    if max(group) >= 5:  # 5인 이상 시 작동할 내용
        if five:
            cv2.rectangle(s_image, (0, 0), (image_w, image_h), (0, 255, 255), 20)
        else:
            cv2.rectangle(s_image, (0, 0), (image_w, image_h), (0, 0, 255), 20)
        five = not five
        print("5인 이상이 감지됐습니다!")
        print("5인 이상이 감지됐습니다!")
        print("5인 이상이 감지됐습니다!")

    return s_image, five


while True:
    screen = np.array(ImageGrab.grab(bbox=(left_top_x, left_top_y, left_top_x+detect_width, left_top_y+detect_height)))
    frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=threshold
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image, Five = draw_bbox(frame, pred_bbox, five=Five)
    result = np.asarray(image)

    cv2.namedWindow("No more 5", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("No more 5", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
