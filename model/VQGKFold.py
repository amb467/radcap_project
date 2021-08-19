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
            
    def __iter__(self):
        self.current_fold = 0
        return self
        
    def __next__(self):
        if len(self.folds) <= self.current_fold:
            raise StopIteration
            
        fold_list = [(i + self.current_fold) % len(self.folds) for i in range(len(self.folds))]
        test_set = self.folds[fold_list.pop()]
        val_set = self.folds[fold_list.pop()]
        train_set = []
        
        for fold in fold_list:
            train_set.extend(self.folds[fold])
        
        self.current_fold += 1      
        return train_set, val_set, test_set