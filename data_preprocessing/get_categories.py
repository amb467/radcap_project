import glob, json, os


def build_image_set(image_dir):   
    image_set = set()
    
    for img_file in glob.iglob(os.path.join(image_dir, "*.jpg")):
        img_file = os.path.basename(img_file)
        image_set.add(img_file)
    
    return image_set
    
with open("instances_train2014.json") as json_file:
    data_set = json.load(json_file)


image_files = build_image_set('/Users/amandawork/Documents/My_Papers/Crowdsourcing_Project/radcap_project/images/')
image_categories = {
    "images": {}
}

for item in data_set["images"]:
    if item["file_name"] in image_files:
        image_categories["images"][item["id"]] = set()

for item in data_set["annotations"]:
    if item["image_id"] in image_categories["images"]:
        image_categories["images"][item["image_id"]].add(item["category_id"])

image_categories["categories"] = data_set["categories"]

for image_id, categories in image_categories["images"].items():
    image_categories["images"][image_id] = list(categories)

with open("image_categories.json", "w") as json_file:
    json.dump(image_categories, json_file, indent = 3)

