import torch
import pdb
import numpy as np
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
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
# from caption import caption_image_beam_search

def caption_image_beam_search(encoder, decoder, image, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    # img = imread(image_path)
    img = image
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    kmax = np.array(complete_seqs_scores)
    kmaxIndices = kmax.argsort()[-k:][::-1]
    kHypotheses = list()
    for ind in kmaxIndices:
        kHypotheses.append(complete_seqs[ind])

    return seq, alphas, kHypotheses



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
        batch_size=1, num_workers=workers, pin_memory=True)
    fileName = 'pseudo_parallel_corpus_'+str(args.beam_size)+'.txt'
    pseudo_parallel_corpus = open(fileName, 'w')
    # recent_bleu4 = validate(val_loader=val_loader,
    #                         encoder=encoder,
    #                         decoder=decoder,
    #                         criterion=criterion)

    # Encode, decode with attention and beam search
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        # if i == 100:
        #     break
        img = imgs[0]
        cap = caps[0]
        seq, alphas, kHypotheses = caption_image_beam_search(encoder, decoder, img, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)
        # pdb.set_trace()
        # print(seq)
        # print(alphas)
        # print("cap", cap)
        # print("kHypotheses", kHypotheses)
        # references = [cap]
        references = []  # references (true captions) for calculating BLEU-4 score
        # for ref in references[0]:
        #     references.append(list(map(lambda w: rev_word_map[w] if rev_word_map[w] not in ['<pad>','<start>'] else '.',ref)))
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
        actual_captions = []
        for ref in references[0]:
            actual_captions.append(list(map(lambda w: rev_word_map[w] if rev_word_map[w] not in ['<pad>','<start>'] else '.', ref)))

        maxBleu = 0
        best_caption = ''
        # corresponding_caption = 
        # print("references", references)
        # pdb.set_trace()
        hypotheses = list()  # hypotheses (predictions)
        for hypothesis in kHypotheses:
            # bleu4 = corpus_bleu(references, [hypotheses], emulate_multibleu=True)
            # pdb.set_trace()
            # hyp_caps = hypothesis[j].tolist()
            hyp_caps = hypothesis
            hyp_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    [hyp_caps]))  # remove <start> and pads
            hypotheses.append(hyp_captions)
            # for j in range(hypothesis.shape[0]):
            #     hyp_caps = hypothesis[j].tolist()
            #     hyp_captions = list(
            #         map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            #             img_caps))  # remove <start> and pads
            #     hypotheses.append(hyp_captions)
            # print("hyp_captions", hyp_captions)
            hyp_captions = hyp_captions*5
            hyp = []
            for h in hyp_captions:
                hyp.append(list(map(lambda w: rev_word_map[w] if rev_word_map[w] not in ['<pad>','<start>'] else '.', h)))
            # print("len of hypotheses", len(hypotheses))
            # bleu = sentence_bleu(references, hypotheses)
            bleu = corpus_bleu(actual_captions, hyp)
            if bleu > maxBleu:
                best_caption = ' '.join(hyp[0][:len(hyp[0])-1])
                # print("best caption - ", best_caption)
                maxBleu = max(maxBleu, bleu)
        if i % args.beam_size == 0:
            print("maxBleu", maxBleu)
            print("best caption - ", best_caption)
            pseudo_parallel_corpus.write(str(i)+'\t'+best_caption+'\n')
