import glob, json, os


def build_image_set(image_dir):   
    image_set = set()
    
    for img_file in glob.iglob(os.path.join(image_dir, "*.jpg")):
        img_file = os.path.basename(img_file)
        image_set.add(img_file)
    
    return image_set
    
def extract_categories(args):

    with open(args.VQG_instances_file) as json_file:
        data_set = json.load(json_file)
        
    image_files = build_image_set(args.image_dir)
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

    with open(args.category_file, "w") as json_file:
        json.dump(image_categories, json_file, indent = 3)

def merge_categories_to_corpora(args):

    # Open the category file
    with open(args.category_file, "r") as json_file:
        image_categories = json.load(json_file)
    
    # For each corpus, go through the corpus and add category information
    for corpus_file in glob.iglob(os.path.join(args.corpus_dir, "*.json")):
        if corpus_file == args.category_file:
            continue
        
        print(f'Found {corpus_file}')
        with open(corpus_file) as json_file:
            corpus = json.load(json_file)
        
        for image_id in corpus.keys():
            categories = image_categories["images"][image_id]
            for item in corpus[image_id]:
                item["categories"] = categories
        
        with open(f'{corpus_file}.new', 'w') as json_file:
            json.dump(corpus, json_file, indent=3)
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get category data')
    parser.add_argument('--image_dir', type=str, default='./images', help='The directory where the image data set is kept.  All images are assumed to be in the format "*_<image id>.jpg"')
    parser.add_argument('--VQG_instances_file', type=str, default='"instances_train2014.json"', help='File from the VQG data set with category information') 
    parser.add_argument('--category_file', type=str, default='./data_sets/image_categories.json', help='The JSON file where LWT category information is stored')
    parser.add_argument('--corpus_dir', type=str, default='./data_sets', help='The directory where JSON data sets should be stored') 
    args = parser.parse_args()
    
    merge_categories_to_corpora(args)
    

