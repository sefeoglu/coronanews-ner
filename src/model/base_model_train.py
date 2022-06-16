
import flair
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import  ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

import argparse


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

    label_dict = corpus.make_tag_dictionary(tag_type)
    print(label_dict)

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    model = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=tag_type,
                            dropout=0.5)


    trainer: ModelTrainer = ModelTrainer(model, corpus)
    print("==================Training=====================")

    trainer.train(out_path,
                    learning_rate=l_rate, 
                    mini_batch_size=BS,
                    max_epochs=epochs,
                    embeddings_storage_mode='cpu',
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