import cv2

def format_detection(object_name, object_id, bbox, subobject_name, subobject_id, subobject_bbox):
    return {
        "object": object_name,
        "id": object_id,
        "bbox": bbox,
        "subobject": {
            "object": subobject_name,
            "id": subobject_id,
            "bbox": subobject_bbox
        }
    }

def save_cropped_image(image, bbox, save_path):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped_image)
