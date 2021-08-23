"""

    The purpose of this file is to input question data from several sources and output
    data sets in JSON format.  The input files can be:
    
    - COCO data files from the VQG data set (https://www.microsoft.com/en-us/download/details.aspx?id=53670)
    - Fill in information here about our corpus

"""
import copy, csv, glob, json, os, pickle, random
from collections import defaultdict

"""
    This method inputs the directory where the image files are stored and returns a
    dictionary object linking image id to file name.  All of the other data sets will
    use this dictionary
"""
def build_image_dictionary(image_dir):
    _check_is_dir(image_dir)    
    image_dict = {}
    
    for img_file in glob.iglob(os.path.join(image_dir, "*.jpg")):
        img_id, img_file = _parse_image_path(img_file)
        image_dict[img_id] = img_file
    
    return image_dict

"""
    Get a list of all image ids in the LWT corpus
"""    
def get_image_list(corpus_file):
    with open(corpus_file, 'r') as corpus_file:
        corpus = json.load(corpus_file)
    return list(corpus.keys())

"""
    Join user and question data based on user id
"""
def join_user_question_data(users, questions): 
    users = _read_csv(users, ['User ID', 'IsBlack', 'IsFemale', 'Is40Plus'])
    questions = _read_csv(questions, ['Question Content','Image ID','User ID'])
    
    user_dict = {}
    for row in users:
        user_id = row['User ID']
        user_dict[user_id] = dict([(header, row[header] == 'Y') for header in ['IsBlack', 'IsFemale', 'Is40Plus']])
        
    for row in questions:
        user_id = row['User ID']
        row.update(user_dict[user_id])
    
    return questions
        
"""
    This data set uses only questions from the original VQG data set (referenced above)
    For the list of corpus images, this method will get the VQG questions for that image
    and output the image id, image file path, and questions in JSON format
    
    The VQG corpus comes in CSV files in the format:
    image_id,image_url,questions,ratings
"""
def generate_control_corpus(img_dict, vqg_dir):
    _check_is_dir(vqg_dir) 
    control_corpus = {}
    
    for vqg_file in glob.iglob(os.path.join(vqg_dir, "*.csv")):
        corpus = _read_vqg_corpus_file(vqg_file)
        corpus = {corpus: corpus[img_id] for img_id in corpus.keys() if img_id in img_dict}
        control_corpus.update(corpus)
    
    return control_corpus        

"""
    For pretraining, we need a selection of images that are not in the LWT corpus.
    
    This method will make two corpora of identical size - one to be used for pretraining
    and the other to be used for validation of the pretrained models.
"""
def generate_pretraining_corpus(image_list, vqg_dir, image_count):
    vqg_file = os.path.join(vqg_dir, "coco_train_all.csv")
    corpus = _read_vqg_corpus_file(vqg_file)
    image_ids = [image_id for image_id in corpus.keys() if image_id not in image_list]
    random.shuffle(image_ids)
    image_ids = image_ids[:(2*image_count)]
    training_ids = set(image_ids [:image_count])
    validation_ids = set(image_ids).difference(training_ids)
    
    return {img_id: corpus[img_id] for img_id in training_ids},  {img_id: corpus[img_id] for img_id in validation_ids}
    
"""
    Filter out a corpus with only the specified demographic
"""

def generate_demographic_corpus(questions, control_corpus, demographic):
    # First, extract corpus information from the Look Who's Talking corpus that matches
    # the demographic
    corpus = defaultdict(list)
 
    for row in questions:
        if not row[demographic]:
            continue
            
        image_id = int(row['Image ID'])
        image_path = control_corpus[image_id][0]['image_path']
        question = row['Question Content'].strip()
        user = row['User ID']
        corpus[image_id].append({'question': question, 'image_path': image_path, 'user': user})

    # Prepare statistics that will be used to ensure that this corpus has the same
    # min qs, max qs, and total qs as the control corpus
    control_min_qs, control_max_qs, control_total_qs = get_min_max_total(control_corpus)
    control_corpus_questions_included = 0
    
    # Make a copy of the control corpus with questions in randomized order per image id
    control_corpus = copy.deepcopy(control_corpus)
    for image_id, q_list in control_corpus.items():
        random.shuffle(q_list)
        control_corpus[image_id] = q_list
    
    # Ensure that this corpus has at least min_qs questions and at most max_qs questions
    # per image.  If there aren't enough questions, take some from the control corpus
    for image_id, q_list in corpus.items():
        
        # If there are too many questions, randomly remove some
        if len(q_list) > control_max_qs:
            random.shuffle(q_list)
            q_list = q_list[:control_max_qs]

        # If there are not enough questions, select some from the control corpus
        while len(q_list) < control_min_qs:
            control_q = control_corpus[image_id].pop()
            q_list.append(control_q)    
            control_corpus_questions_included += 1
    
        # It should now be the case that control_min_qs <= len(q_list) <= control_max_qs
        corpus[image_id] = q_list
        
    # If the total corpus has fewer questions than the control corpus, add questions
    # from the control corpus until it matches 
    min_qs, max_qs, total_qs = get_min_max_total(corpus)
    fill_in_count = control_total_qs - total_qs
    fill_in_image_ids = list(control_corpus.keys())
    
    while 0 < fill_in_count:
        image_id = random.choice(fill_in_image_ids)
        if not image_id in corpus:
            corpus[image_id] = []
        
        # If this image has fewer than the max number of questions, add one from the
        # control corpus
        if len(corpus[image_id]) < control_max_qs:
            control_q = control_corpus[image_id].pop()
            corpus[image_id].append(control_q)
            control_corpus_questions_included += 1
            fill_in_count -= 1
            
        # If this image has the max number of questions already, remove it from the
        # fill in list
        else:
            fill_in_image_ids.remove(image_id)  
    
    return corpus, control_corpus_questions_included

"""
    Output the dictionary object as a JSON file
"""
def output_corpus(output_dir, corpus_name, dict_obj):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    outfile = os.path.join(output_dir, f'{corpus_name}.json')
    with open(outfile, 'w') as outfile:
        json.dump(dict_obj, outfile, indent = 3)

"""
    Get the minimum number of questions per image, the maximum number of questions per
    image, and the total number of questions in the corpus
"""
def get_min_max_total(corpus):
    num_qs = [len(q_list) for q_list in corpus.values()]
    return min(num_qs), max(num_qs), sum(num_qs)           
                    
"""
    Check if the passed string is a directory.  If not, raise an Exception
"""
def _check_is_dir(dir_str):
    if not os.path.isdir(dir_str):
        raise Exception(f'Not a directory: {dir_str}')

"""
    Convert a VQG image path URL into just the file name and image id
"""
def _parse_image_path(img_path):
        img_path = os.path.basename(img_path)
        img_id = int(img_path.split(".")[-2].split("_")[-1])
        return img_id, img_path
        
"""
    Open and read CSV file, return a list of dictionary items
"""
def _read_csv(csv_file, header_list):
    row_data = []
    with open(csv_file, 'r', encoding='utf-8-sig') as csv_file:
        for row in csv.DictReader(csv_file):
            row_data.append(dict([(header, row[header]) for header in header_list]))
    return row_data
    
"""
    Convert a VQG corpus file into a corpus dictionary
"""
def _read_vqg_corpus_file(vqg_file):  
    corpus = {}
    
    for row in _read_csv(vqg_file, ['image_id', 'image_url', 'questions']):
        img_id = int(row['image_id'])
        _, img_file = _parse_image_path(row['image_url'])           
        questions = row['questions'].split('---')
        corpus[img_id] = [{'question': question, 'image_path': img_file} for question in questions]
    
    return corpus   

           
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create data set JSON files for the Look Who\'s Talking Data Set')
    parser.add_argument('--image_dir', type=str, default='./images', help='The directory where the image data set is kept.  All images are assumed to be in the format "*_<image id>.jpg"')
    parser.add_argument('--output_dir', type=str, default='./data_sets', help='The directory where JSON data sets should be stored')  
    parser.add_argument('--vqg_dir', type=str, default='./data_preprocessing/vqg_files', help='The directory with the VQG csv files') 
    parser.add_argument('--users', type=str, default='./data_preprocessing/users.csv', help='The csv file with Prolific user information')
    parser.add_argument('--questions', type=str, default='./data_preprocessing/questions.csv', help='The csv file with Prolific question information')
    parser.add_argument('--pretraining_image_count', default=2000, type=int, help='The number of images to select for ')
  
    args = parser.parse_args()

    image_list = get_image_list(os.path.join(args.output_dir, 'control.json'))
    """    
    image_dict = build_image_dictionary(args.image_dir)
    
    questions = join_user_question_data(args.users, args.questions)
    corpus_labels = {'black_dominated': 'IsBlack', 'female_dominated': 'IsFemale', '40plus_dominated': 'Is40Plus'}
    corpora = {}
    control_qs_included = {}
    
    corpora['control'] = generate_control_corpus(image_dict, args.vqg_dir)
    
    for label, header in corpus_labels.items():
        corpora[label], control_qs_included[label] =  generate_demographic_corpus(questions, corpora['control'], header)

    for label, corpus in corpora.items():
        output_corpus(args.output_dir, label, corpus)
        min_qs, max_qs, total_qs = get_min_max_total(corpus)
        
        print(f'Created corpus {label} with min number of questions {min_qs} and max number of questions {max_qs} and total questions {total_qs}.')
        
        if label in control_qs_included:
            print(f'Questions from control corpus: {control_qs_included[label]}')   
    """
    
    pretraining, validation = generate_pretraining_corpus(image_list, args.vqg_dir, args.pretraining_image_count)
    
    image_count = len(pretraining.keys())
    question_count = sum([len(q_list) for q_list in pretraining.values()])
    output_corpus(args.output_dir, "pretraining", pretraining)
    print(f'Created pretraining corpus with {image_count} images and {question_count} questions')
    
    image_count = len(validation.keys())
    question_count = sum([len(q_list) for q_list in validation.values()])
    output_corpus(args.output_dir, "pretraining_validation", validation)
    print(f'Created pretraining validation corpus with {image_count} images and {question_count} questions')
    
    
