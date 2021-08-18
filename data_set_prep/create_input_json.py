"""

    The purpose of this file is to input question data from several sources and output
    data sets in JSON format.  The input files can be:
    
    - COCO data files from the VQG data set (https://www.microsoft.com/en-us/download/details.aspx?id=53670)
    - Fill in information here about our corpus

"""
import csv, glob, json, os, random
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
        img_file = os.path.basename(img_file)
        img_id = int(img_file.split(".")[-2].split("_")[-1])
        image_dict[img_id] = img_file
    
    return image_dict
    
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
        for row in _read_csv(vqg_file, ['image_id', 'questions']):
            img_id = int(row['image_id'])
            if not img_id in img_dict:
                continue
            
            questions = row['questions'].split('---')
            control_corpus[img_id] = [{'question': question, 'image_path': img_dict[img_id]} for question in questions]
    
    return control_corpus        

"""
    Construct corpora for each demographic.  Each corpus will have the same total number
    of questions as the control corpus and each image will have a number of questions
    ranging from the min to the max questions per image from the control corpus
    
    users: a CSV file containing information about each user
    question: a CSV file containing information about the questions produced by each user
    control_corpus: The control corpus with questions from the VQG data set
    min_qs: The minimum number of questions per image in the control corpus
    max_qs: The maximum number of questions per image in the control corpus
    total_qs: The total number of questions per image in the control corpus
    
"""
def generate_demographic_corpora(users, questions, control_corpus):
    questions = _join_user_question_data(users, questions)

    min_qs, max_qs, total_qs = _get_min_max_total(control_corpus)
    print(f'Created control corpus with min number of questions {min_qs} and max number of questions {max_qs} and total questions {total_qs}')
    
    bdc = _generate_demographic_corpus(questions, control_corpus, 'IsBlack')
    fdc = _generate_demographic_corpus(questions, control_corpus, 'IsFemale')
    fpdc = _generate_demographic_corpus(questions, control_corpus, 'Is40Plus')
    return bdc, fdc, fpdc
         
"""
    Check if the passed string is a directory.  If not, raise an Exception
"""
def _check_is_dir(dir_str):
    if not os.path.isdir(dir_str):
        raise Exception(f'Not a directory: {dir_str}')

"""
    Output the dictionary object as a JSON file
"""
def _output_corpus(output_dir, corpus_name, dict_obj):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    outfile = os.path.join(output_dir, f'{corpus_name}.json')
    with open(outfile, 'w') as outfile:
        json.dump(dict_obj, outfile)

"""
    Get the minimum number of questions per image, the maximum number of questions per
    image, and the total number of questions in the corpus
"""
def _get_min_max_total(corpus):
    num_qs = [len(q_list) for q_list in corpus.values()]
    return min(num_qs), max(num_qs), sum(num_qs)        

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
    Join user and question data based on user id
"""
def _join_user_question_data(users, questions): 
    users = _read_csv(users, ['ID', 'IsBlack', 'IsFemale', 'Is40Plus'])
    questions = _read_csv(questions, ['Question Content','Image','User'])
    
    user_dict = {}
    for row in users:
        user_id = row['ID']
        user_dict[user_id] = dict([(header, row[header] == 'Y') for header in ['IsBlack', 'IsFemale', 'Is40Plus']])
        
    for row in questions:
        user_id = row['User']
        row.update(user_dict[user_id])
    
    return questions;

"""
    Filter out a corpus with only the specified demographic
"""

def _generate_demographic_corpus(questions, control_corpus, demographic):
    corpus = defaultdict(list)
    min_qs, max_qs, total_qs = _get_min_max_total(control_corpus)
    
    for row in questions:
        is_demographic = row[demographic]
        if not is_demographic:
            continue
            
        image_id = int(row['Image'])
        image_path = control_corpus[image_id][0]['image_path']
        question = row['Question Content'].strip()
        user = row['User']
        corpus[image_id].append({'question': question, 'image_path': image_path, 'user': user})
    
    control_corpus_questions_count = 0
    
    for image_id, q_list in corpus.items():
        if len(q_list) > max_qs:
            q_list = [q_dict for q_dict in q_list if q_dict['user'].startswith('f')]
        
        if len(q_list) > max_qs:
            random.shuffle(q_list)
            q_list = q_list[:max_qs]

        if len(q_list) < min_qs:
            needed_qs = min_qs - len(q_list)
            control_corpus_qs = control_corpus[image_id]
            random.shuffle(control_corpus_qs)
            control_corpus_questions_count += needed_qs
            q_list.extend(control_corpus_qs[:needed_qs])
    
        corpus[image_id] = q_list
        
    min_qs, max_qs, total_qs = _get_min_max_total(corpus)
    print(f'Created corpus {demographic} with min number of questions {min_qs} and max number of questions {max_qs} and total questions {total_qs}.  Questions from control corpus: {control_corpus_questions_count}')
    return corpus
            
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Create data set JSON files for the Look Who\'s Talking Data Set')
    parser.add_argument('--image_dir', type=str, default='../raw_images', help='The directory where the image data set is kept.  All images are assumed to be in the format "*_<image id>.jpg"')
    parser.add_argument('--output_dir', type=str, default='data_sets', help='The directory where JSON data sets should be stored')  
    parser.add_argument('--vqg_dir', type=str, default='vqg_files', help='The directory with the VQG csv files') 
    parser.add_argument('--users', type=str, default='users.csv', help='The csv file with Prolific user information')
    parser.add_argument('--questions', type=str, default='questions.csv', help='The csv file with Prolific question information')
    args = parser.parse_args()
            
    image_dict = build_image_dictionary(args.image_dir)
    control_corpus = generate_control_corpus(image_dict, args.vqg_dir)
    _output_corpus(args.output_dir, 'control', control_corpus)
    
    bdc, fdc, fpdc = generate_demographic_corpora(args.users, args.questions, control_corpus)
    
    