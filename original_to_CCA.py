from multiprocessing import Pool
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import numpy as np
import math
import cv2
import os

# euclidan distance between two points
def euclidean_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) + ((b[1] - a[1]) * (b[1] - a[1])))

# given an image and a detected face, crops image to be 1.5 times bigger then bounding box of the detected face
def crop(img, detected_face):
    image_height, image_width, _ = img.shape

    x1, y1, width, height = detected_face["box"]
    x2, y2 = x1+width, y1+height
    x1_adjust = x1 - ((x2-x1)//2)
    y1_adjust = y1 - ((y2-y1)//2)
    x2_adjust = x2 + ((x2-x1)//2)
    y2_adjust = y2 + ((y2-y1)//2)

    x1_adjust = max(0, x1_adjust)
    y1_adjust = max(0, y1_adjust)
    x2_adjust = min(image_width, x2_adjust)
    y2_adjust = min(image_height, y2_adjust)

    cropped_face = img[y1_adjust:y2_adjust, x1_adjust:x2_adjust]

    landmarks = []
    landmarks.append(list(detected_face["keypoints"]["left_eye"]))
    landmarks.append(list(detected_face["keypoints"]["right_eye"]))

    standard_landmarks = []
    for landmark in landmarks:
        standard_x = landmark[0] - x1_adjust
        standard_y = landmark[1] - y1_adjust
        standard_landmarks.append([standard_x, standard_y])

    return cropped_face, standard_landmarks[0], standard_landmarks[1]

class CCA_all_images:

    def __init__(self, path_to_class_folders, path_to_CCA_save_location):
        self.dataset_path = path_to_class_folders
        self.save_path = path_to_CCA_save_location
        
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.name_list = os.listdir(self.dataset_path)
        self.parallel_CCA()
    
# multi-process CCA method, each process gets 1 image, and saves a new image for each face found
    def parallel_CCA(self):

        for name in self.name_list:
            if(not os.path.isdir(self.save_path + name)):
                os.mkdir(self.save_path + name)
            image_names = os.listdir(self.dataset_path + name)

            source_paths = [f"{self.dataset_path+name}/{i}" for i in image_names]
            # remove .jpg ending for each destination
            destination_paths = [f"{self.save_path + name}/{i[:-4]}" for i in image_names]
            function_params = zip(source_paths, destination_paths)
            with Pool() as p:
                p.starmap(self.Face_Alignment_Wrapper, function_params)

    # finds all faces in an image, and CCA each face
    def Face_Alignment_Wrapper(self, source_path, destination_path):
        detector = MTCNN()
        img = cv2.imread(source_path)

        try:
            results = detector.detect_faces(img)

            for index, face in enumerate(results):
                output_centered_face = self.Face_Alignment(img, face)

                cv2.imwrite(f"{destination_path}_{index}.jpg", output_centered_face)

        except Exception as error:
            print(error)

    # finds the crop of that face, then rotates the image so that the eyes are at the same y cordinate
    def Face_Alignment(self, image, face):
        img_raw = image.copy()

        cropped_image, left_eye, right_eye = crop(img_raw, face)

        cropped_height, cropped_width, _ = cropped_image.shape
        cropped_center = (cropped_height // 2, cropped_width // 2)

        # center of right eye
        right_eye_x = right_eye[0]
        right_eye_y = right_eye[1]

        # center of left eye
        left_eye_x = left_eye[0]
        left_eye_y = left_eye[1]

        exp_center_x = ((right_eye_x + left_eye_x) // 2)
        exp_center_y = ((left_eye_y + right_eye_y) // 2)

        if not left_eye_y == right_eye_y:

            if left_eye_y > right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate image direction to clock
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock

            # Calculated third point (RED)
            a = euclidean_distance([left_eye_x, left_eye_y],
                                        point_3rd)
            b = euclidean_distance([right_eye_x, right_eye_y],
                                        point_3rd)
            c = euclidean_distance([right_eye_x, right_eye_y],
                                        [left_eye_x, left_eye_y])
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = (np.arccos(cos_a) * 180) / math.pi

            if direction == -1:
                angle = 90 - angle
                angle = -angle
            else:
                angle = angle

            rotation_matrix = cv2.getRotationMatrix2D(cropped_center, angle, 1.0)
            aligned_image = cv2.warpAffine(cropped_image, rotation_matrix, [cropped_width, cropped_height],
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            
            # Center of aligned image (RED)
            aligned_height, aligned_width, _ = aligned_image.shape

        else:
            aligned_height, aligned_width, _ = cropped_image.shape

            aligned_image = np.array(cropped_image)

        distance_to_top = exp_center_y
        distance_to_bottom = aligned_height - exp_center_y
        distance_to_left = exp_center_x
        distance_to_right = aligned_width - exp_center_x

        top = max(0, distance_to_bottom - distance_to_top)
        bottom = max(0, distance_to_top - distance_to_bottom)
        left = max(0, distance_to_right - distance_to_left)
        right = max(0, distance_to_left - distance_to_right)

        centered_image = cv2.copyMakeBorder(aligned_image, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

        return centered_image