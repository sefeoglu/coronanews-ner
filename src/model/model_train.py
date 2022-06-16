
import flair
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import  ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings,ELMoEmbeddings,BertEmbeddings

import argparse
import torch
import gc

def getBertEmbeddings():
    flair_forward_embedding = FlairEmbeddings('mix-forward')
    flair_backward_embedding = FlairEmbeddings('mix-backward')

    bert_embedding = BertEmbeddings('bert-base-cased')
    embedding_types = [flair_forward_embedding,flair_backward_embedding, bert_embedding]
    return embedding_types

def getElmoEmbeddings():
    embedding_types = [
        ELMoEmbeddings('small')]
    return embedding_types

def getFlairEmbeddings(type="mix"):
    if type == "mix":
        embedding_types = [
            WordEmbeddings('glove'),
            FlairEmbeddings('mix-forward', chars_per_chunk=128),
            FlairEmbeddings('mix-backward', chars_per_chunk=128)
            ]
    else:
        embedding_types = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward-fast', chars_per_chunk=128),
            FlairEmbeddings('news-backward-fast', chars_per_chunk=128)
            ]

    return embedding_types

def getSimpleWordEmbeddings():

    embedding_types = [
        WordEmbeddings('glove')]

    return embedding_types



def train(corpus_path, out_path, l_rate, BS, epochs, down_sampling):
    columns = {0: 'text', 1: 'ner'}
    tag_type = 'ner'

    corpus: Corpus = ColumnCorpus(
                        corpus_path,
                        column_format=columns,
                        train_file='train.txt',
                        test_file='test.txt',
                        dev_file='dev.txt',
                        tag_to_bioes=tag_type
                    ).downsample(down_sampling)
    max_tokens = 20
    corpus._train = [x for x in corpus.train if len(x) < max_tokens]
    corpus._dev = [x for x in corpus.dev if len(x) < max_tokens]
    corpus._test = [x for x in corpus.test if len(x) < max_tokens]

    label_dict = corpus.make_tag_dictionary(tag_type)


    print(label_dict)

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = getFlairEmbeddings('new')

    embeddings = StackedEmbeddings(embeddings=embedding_types)
    ###
    gc.collect()
    torch.cuda.empty_cache()
    # 5. initialize sequence tagger
    model = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=tag_type,
                            use_crf=True,
                            locked_dropout=0.0, word_dropout=0.0)

   
    trainer: ModelTrainer = ModelTrainer(model, corpus)
    print("==================Training=====================")

    trainer.train(out_path,
                    learning_rate=l_rate, 
                    mini_batch_size=BS,
                    max_epochs=epochs,
                    embeddings_storage_mode='cpu',
                    shuffle=False,
                    write_weights=True,
                    mini_batch_chunk_size=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optional app description')
    # Required positional argument

    parser.add_argument('input', type=str,
                        help='input path')
    parser.add_argument('output', type=str,
                        help='output path')
    parser.add_argument('l_rate', type=float,
                        help='learning_rate')
    parser.add_argument('batch', type=int,
                        help='mini batch size')
    parser.add_argument('epochs', type=int,
                        help='max epochs')
    parser.add_argument('down_sampling', type=float,
                        help='down sampling rate')
        
    # File positional argument
    args = parser.parse_args()

    input_path = args.input
    out_path = args.output
    learning_rate = float(args.l_rate)
    BS = int(args.batch)
    epochs = int(args.epochs)
    down_sampling = float(args.down_sampling)
    
    train(input_path, out_path, learning_rate, BS, epochs, down_sampling)