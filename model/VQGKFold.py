import random

# Creates folds based on images
class VQGKFold():

    """
        folds: the number of folds
        image_ids: the list of image ids that will be divided into folds
    """
    def __init__(self, folds, image_ids):
        random.shuffle(image_ids)
        self.folds = [[] for i in range(folds)]
        for i, image_id in enumerate(image_ids):
            self.folds[i % folds].append(image_id)
    
    """
        Get the splits for the passed fold
    """
    def get(self, fold):
        fold_list = [(i + fold) % len(self.folds) for i in range(len(self.folds))]
        test_set = self.folds[fold_list.pop()]
        val_set = self.folds[fold_list.pop()]
        train_set = flatten([self.folds[i] for i in fold_list])            
        return train_set, val_set, test_set
    
    """
        Return a list of question objects corresponding to the image ids in the split
    """
    def get_data_for_split(image_ids, corpus):
        return flatten([corpus[image_id] for image_id in image_ids])
                           
    def __iter__(self):
        self.current_fold = 0
        return self
        
    def __next__(self):
        if len(self.folds) <= self.current_fold:
            raise StopIteration
            
        train_set, val_set, test_set = self.get(self.current_fold)
        self.current_fold += 1      
        return train_set, val_set, test_set

def flatten(list_of_lists):
    return [item for this_list in list_of_lists for item in this_list]
