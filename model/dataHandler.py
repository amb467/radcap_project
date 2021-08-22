import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import os
import pickle
from PIL import Image
from model.VocabularyFromPreTrained import *

data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class VQGDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, data_set, vocab, transform):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory
            data_set: JSON data set file
            vocab: Vocabulary object
            transform: image transformer
        """

        self.root = root
        self.data_set = data_set
        self.vocab = vocab
        self.transform = transform
    	
    def __getitem__(self, index):
        """Returns one data pair (image and question)."""

        image = os.path.join(self.root, self.data_set[index]['image_path'])

        if os.path.exists(image):
            image = Image.open(image).convert('RGB')
        else:
            raise Exception(f'VQGDataset.__getitem__: no such image: {img_file_path}')

        image = self.transform(image) if self.transform else image

        # Convert caption (string) to word ids
        target = self.data_set[index]['question']
        target = self.vocab.sentence_to_idx(target)

        return image, torch.Tensor(target)

    def __len__(self):
        return len(self.data_set)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption)
            - image: torch tensor of shape (3, 256, 256)
            - caption: torch tensor of shape (?); variable length
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list; valid length for each padded caption
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, questions = zip(*data)

    images = torch.stack(images, 0)
    #images_tensor = torch.zeros(len(images), 3, 224, 224)
    #for i, image in enumerate(images):
    #    images_tensor[i, :, :, :] = image

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(question) for question in questions]
    targets = torch.zeros(len(questions), max(lengths)).long()
    for i, question in enumerate(questions):
        end = lengths[i]
        targets[i, :end] = question[:end]

    return images, targets, lengths

def get_loader(root, data_set, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Look Who's Talking dataset."""

    transform = data_transform if not transform else transform
    data_set = VQGDataset(root, data_set, vocab, transform)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test out data handler')
    parser.add_argument('--data_set_dir', type=str, default='data_sets/', help='The directory with data set and vocab files')
    parser.add_argument('--root_dir', type=str, default='images/', help='The directory with the image files')
    args = parser.parse_args()

    labels = ['control', '40plus_dominated', 'black_dominated', 'female_dominated']

    for label in labels:
        data_set_file = os.path.join(args.data_set_dir, f'{label}.json')
        vocab_file = os.path.join(args.data_set_dir, f'{label}.vocab.pkl')
        data_loader = get_loader(args.root_dir, data_set_file, vocab_file, None, 4, True, 2)
    
        with open(vocab_file, 'rb') as vocab:
            vocab = pickle.load(vocab)
            
        i = 0
    
        for images, targets, lengths in data_loader:
            i = i + 1
            if i > 2:
                break
        
            print(f'Images: {images}')
        
            targets = [vocab.idxs_to_sentence(target.tolist()) for target in targets]
            print(f'Targets: {targets}')
            print(f'Lengths: {lengths}')
