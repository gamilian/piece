import cv2
import os
import numpy as np
import config
import copy
from coutours_process import get_coutours

import cv2
import numpy as np



def crop_image(image_root, crop_image_name, bg_color):
    image_path = os.path.join(image_root, crop_image_name)
    print(image_path)
    # Load the image
    image = cv2.imread(image_path)
    # image[2187:2187+130, 435:435+187] = bg_color
    
    # Find contours of the image
    contours = get_coutours(image, bg_color)

    # Iterate over contours and crop the pieces
    for idx, contour in enumerate(contours):
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Avoid cutting into other pieces
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255,255, 255), thickness=cv2.FILLED)
        tmp = copy.deepcopy(image)
        tmp[np.where((mask != [255,255, 255]).all(axis=2))] = bg_color
        # Crop the image using the bounding box
        crop = tmp[y:y+h, x:x+w]
        
        # Prepare file path for saving the image
        file_path = os.path.join(image_root, f"fragment_{str(idx+1).rjust(4, '0')}.png")
        
        # Save the cropped piece
        cv2.imwrite(file_path, crop)

if __name__ == '__main__':
    crop_image(config.dataset_path, config.crop_image_name, config.bg_color)
