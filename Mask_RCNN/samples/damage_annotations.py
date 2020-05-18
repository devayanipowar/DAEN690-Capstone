import glob
import json

from pycocotools import mask as maskUtils


class Damage_Assessment_Annotation(object):


    info = {
        "description": "xview2 dataset",
        "url": "https://xview2.org/"
    }
    images = []
    annotations = []
    categories = [
        {
            "supercategory": "building",
            "id": 0,
            "name": "no-damage"
        },
        {
            "supercategory": "building",
            "id": 1,
            "name": "minor-damage"
        },
        {
            "supercategory": "building",
            "id": 2,
            "name": "major-damage"
        },
        {
            "supercategory": "building",
            "id": 3,
            "name": "destroyed"

        }

    ]
    image_uid_to_image_id = {}
    annotation_uid_to_annotation_id = {}

    class_to_index_mappings = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3
    }

    def __init__(self):

        self.image_count = 0
        self.annotation_count = 0
        self.images = []
        self.annotations = []

    def add_image_from_filename(self, filename):
        self.image_count += 1
        with open(filename, "r") as myfile:
            json_string = u'{}'.format(myfile.read())
        json_data = json.loads(json_string)
        width = int(json_data["metadata"]["width"])
        height = int(json_data["metadata"]["height"])
        image_uid = json_data["metadata"]["id"]
        image_id = self.image_count
        self.image_uid_to_image_id[image_uid] = image_id

        image_data = {
            "file_name": json_data["metadata"]["img_name"],
            "height": height,
            "width": width,
            "id": image_id
        }
        self.images.append(image_data)

        class_to_index_mappings = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3
        }

        xy_features = json_data["features"]["xy"]
        for xy_feature in xy_features:
            if xy_feature["properties"]["subtype"] != u'un-classified':
                polygon_text = xy_feature["wkt"]
                annotation_uid = xy_feature["properties"]["uid"]

                polygon_values = polygon_text[
                                 polygon_text.find("((") + 2:-2].replace(",", "").split(" ")

                polygon = []
                x_coords = []
                y_coords = []
                for i in range(0, len(polygon_values), 2):
                    x, y = float(polygon_values[i]), float(polygon_values[i + 1])
                    x_coords.append(x)
                    y_coords.append(y)
                    polygon.append(x)
                    polygon.append(y)

                bounding_box_width = max(x_coords) - min(x_coords)
                bounding_box_height = max(y_coords) - min(y_coords)
                bounding_box = [
                    float(min(x_coords)),
                    float(min(y_coords)),
                    float(bounding_box_width),
                    float(bounding_box_height)
                ]

                # height and width from original image
                rle = maskUtils.frPyObjects([polygon], height, width)
                area = maskUtils.area(rle)[0]

                self.annotation_count += 1
                self.annotation_uid_to_annotation_id[annotation_uid] = self.annotation_count
                annotation_data = {
                    "segmentation": [polygon],
                    "iscrowd": 0,
                    "image_id": image_id,
                    "category_id": class_to_index_mappings[xy_feature["properties"]["subtype"]],
                    "id": self.annotation_count,
                    "bbox": bounding_box,
                    "area": float(area)
                }
                self.annotations.append(annotation_data)


    def write_to_json(self, filename):
        data = {
            "info": self.info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)


if __name__ == "__main__":
    formatter = Damage_Assessment_Annotation()
    annotation_files = glob.glob("C:/Users/dpawa/OneDrive/Documents/project690/xBD/labels/*")
    for filename in annotation_files:
        formatter.add_image_from_filename(filename)
    formatter.write_to_json("coco_format_damage.json")