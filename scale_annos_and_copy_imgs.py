import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Jacob Nielsen, Jonathan Lorray, Srijan Das, Peter Schneider-Kamp, Mubarak Shah and Aritra Dutta
info = {
    "description": "Lowes Self Checkout Dataset",
    "url": "N/A",
    "version": "1.0",
    "year": 2023,
    "contributor": ""
}

licences = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "No License"
    }
]


def scale_images_and_annotations(ann_file, image_dir, output_dir, new_img_width=800):
    # Load COCO annotations
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Create a new COCO annotation object
    scaled_coco_data = {
        "info": info,  # coco_data["info"],
        "licenses": [],  # coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    prev_scale = 0
    # Iterate over images
    for image_info in tqdm(coco_data["images"], desc="Processing Images"):
        image_file = find_image_file(image_info["file_name"], image_dir)
        if image_file is None:
            continue

        # Load image
        image = cv2.imread(image_file)

        # print(image_info)
        # Calculate scaling factor
        # we take these from the annotation file
        anno_width = image_info['width']  # 600 # image.shape[1]
        anno_height = image_info['height']  # 337 # image.shape[0]

        width = image.shape[1]
        height = image.shape[0]
        # print("img width: ", width)
        # print("img height: ", height)

        """
        Section If we want to scale:
        """
        # # Scale image
        if new_img_width is not None:
            new_width = new_img_width  # THIS SHOULD BE DEPENDENT ON THE SRC image full scale
            scale = height / width  # width / height
            if scale != prev_scale:
                print("scale: ", scale)
                prev_scale = scale
            new_height = int(scale * new_width)  # int(image.shape[0] * scale)

            # print("saving image with, width: ", width, " ,height: ", new_height)
            scaled_image = cv2.resize(image, (new_width, new_height))

            # Save scaled image
            scaled_image_file = os.path.join(output_dir, image_info["file_name"])
            cv2.imwrite(scaled_image_file, scaled_image)

        """
        Section if we dont wan't to scale
        """
        if new_img_width is None:
            new_width = width
            new_height = height
            dst_img_path = os.path.join(output_dir, image_info["file_name"])
            cv2.imwrite(dst_img_path, image)

        # Update image info
        image_info["width"] = new_width
        image_info["height"] = new_height

        # scale the annotations to match the image
        x_scale = new_width / anno_width
        y_scale = new_height / anno_height

        # Update annotations
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                bbox = annotation["bbox"]
                scaled_bbox = [
                    bbox[0] * x_scale,
                    bbox[1] * y_scale,
                    bbox[2] * x_scale,
                    bbox[3] * y_scale
                ]
                annotation["bbox"] = scaled_bbox

                scaled_coco_data["annotations"].append(annotation)

        scaled_coco_data["images"].append(image_info)

    # Save scaled COCO annotations
    scaled_ann_file = os.path.splitext(ann_file)[0] + "_1600_scaled.json"
    # scaled_ann_file = os.path.splitext(os.path.basename(ann_file)[0]) + "_scaledup.json"
    with open(scaled_ann_file, "w") as f:
        json.dump(scaled_coco_data, f, indent=4)

    print("Scaled annotation file saved:", scaled_ann_file)


def find_image_file(filename, directory):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


# Example usage
# VAL
annotation_file = "/data/jlorray1/dvd/labelled/supervised_annotations/aerial/aerial_test.json"
image_directory = "/data/jlorray1/dvd/labelled/test/aerial/"
output_directory = "scale_down_1600/test"  # image output

# annotation_file = "/data/jlorray1/dvd/labelled/supervised_annotations/aerial/aerial_train.json"
# image_directory = "/data/jlorray1/dvd/labelled/train/aerial/"
# output_directory = "scale_down_1600/train/aerial"  # image output

# annotation_file = "../../SDU-GAMODv4-old/supervised_annotations/ground/aligned_ids/ground_valid_aligned_ids_w_indicator.json"
# image_directory = "../../Everything_to_Jacob/datasetV3/annotated_dataset_full_size/"
# output_directory = "../PUBLICATION/labelled_dataset/val/ground" # image putput

scale_images_and_annotations(annotation_file, image_directory, output_directory, new_img_width=1600)
