import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import os
from PIL import Image
from build_vocab import Vocabulary

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
        self.vocab = vocab
        self.transform = transform

        with open(data_set, 'r') as data_set:
            data_set = json.load(data_set)

        self.data_set = []
        for question_list in data_set.values():
            self.data_set.extend(question_list)

        #print(self.data_set)

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


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def make_weights_for_balanced_classes(json_imgs, nclasses):
    count = [0] * nclasses
    for item in json_imgs:
        if item['fracture'] == str(1): #hard coded classes '0' and '1'
            count[1] += 1
        else:
            count[0] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(json_imgs)

    for idx, img in enumerate(json_imgs):
        if img['fracture'] == 1:
            weight[idx] = weight_per_class[1]
        else:
            weight[idx] = weight_per_class[0]
    return weight


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

    import argparse, pickle, random

    parser = argparse.ArgumentParser(description='Test out data handler')
    parser.add_argument('--data_set', type=str, help='The data set in JSON format')
    parser.add_argument('--vocab', type=str, help='The pickled vocabulary object')
    parser.add_argument('--root_dir', type=str, default='raw_images/', help='The directory with the image files')
    args = parser.parse_args()

    with open(args.vocab, 'rb') as vocab:
        vocab = pickle.load(vocab)

    data_loader = get_loader(args.root_dir, args.data_set, vocab, None, 4, True, 2)
    
    i = 0
    
    for images, targets, lengths in data_loader:
        i = i + 1
        if i > 2:
            break
        
        print(f'Images: {images}')
        
        targets = [vocab.idxs_to_sentence(target.tolist()) for target in targets]
        print(f'Targets: {targets}')
        print(f'Lengths: {lengths}')



