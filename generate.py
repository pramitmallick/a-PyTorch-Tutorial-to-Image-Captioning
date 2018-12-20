import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim
import numpy as np
import json
import torchvision.transforms as transforms
from models import Encoder, DecoderWithAttention
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from caption import caption_image_beam_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_folder = '../data/coco'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    # parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)

    # recent_bleu4 = validate(val_loader=val_loader,
    #                         encoder=encoder,
    #                         decoder=decoder,
    #                         criterion=criterion)

    # Encode, decode with attention and beam search
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        if i == 10:
            break
        img = imgs[0]
        cap = caps[0]
        seq, alphas, seqs = caption_image_beam_search(encoder, decoder, img, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)
        # print(seq)
        # print(alphas)
        print("cap", cap)
        print("seqs", seqs)
        references = [cap]
        maxBleu = 0
        for hypotheses in seqs:
            bleu4 = corpus_bleu(references, [hypotheses], emulate_multibleu=True)
            maxBleu = max(maxBleu, bleu4)
        print("maxBleu", maxBleu)

