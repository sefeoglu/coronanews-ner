
import flair
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import  ColumnCorpus
import torch
import argparse
import gc

gc.collect()

torch.cuda.empty_cache()

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
    max_tokens = 10
    corpus._train = [x for x in corpus.train if len(x) < max_tokens]
    corpus._dev = [x for x in corpus.dev if len(x) < max_tokens]
    corpus._test = [x for x in corpus.test if len(x) < max_tokens]
    
    tagger = SequenceTagger.load("flair/ner-english-ontonotes")
    state = tagger._get_state_dict()

    tag_dictionary = corpus.make_tag_dictionary(tag_type)


    state['tag_dictionary'] = tag_dictionary
    START_TAG: str = "<START>"
    STOP_TAG: str = "<STOP>"

    state['state_dict']['transitions'] = torch.nn.Parameter(torch.randn(len(tag_dictionary), len(tag_dictionary)))
    state['state_dict']['transitions'].detach()[tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
    state['state_dict']['transitions'].detach()[:, tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000
    num_directions = 2 if tagger.bidirectional else 1
    linear_layer =  torch.nn.Linear(tagger.hidden_size * num_directions, len(tag_dictionary))
    state['state_dict']['linear.weight'] = linear_layer.weight
    state['state_dict']['linear.bias'] = linear_layer.bias

    model = SequenceTagger._init_model_with_state_dict(state)

    trainer: ModelTrainer = ModelTrainer(model, corpus)
    print("==================Training=====================")

    trainer.train(out_path,
                    learning_rate=l_rate, 
                    mini_batch_size=BS,
                    max_epochs=epochs,
                    embeddings_storage_mode='none',
                    shuffle=False,
                    write_weights=True)

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