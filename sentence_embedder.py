import string

import argparse

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sentence_transformers import SentenceTransformer


def sentence_processor(file_path):
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            line= line.strip()
            line = line.translate(str.maketrans('','', string.punctuation))
            tokens = word_tokenize(line)
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            preprocessed_sentence = ' '.join(tokens)
            preprocessed_sentences.append(preprocessed_sentence)
    return preprocessed_sentences


def sentence_encoder(model, preprocessed_sentences):
    model = SentenceTransformer(model)
    list_sentence = []
    
    for sentence in preprocessed_sentences:
        vec = model.encode(sentence)
        list_sentence.append(vec)
    return list_sentence

# load embeddings as matrix
def load_sentence_embeddings(file_path):
    with open(file_path, 'r') as file:
        embeddings = []
        for line in file:
            vec = [float(i) for i in line.strip().split()]
            embeddings.append(vec)
    return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stackoverflow',
                            choices=['stackoverflow', 'biomedical', 'searchsnippets'])
    parser.add_argument('--output_file', default='embeddings.txt')
    parser.add_argument('--model', default="paraphrase-MiniLM-L6-v2", 
                        choices=['paraphrase-multilingual-MiniLM-L12-v2','paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                        type=str)
    
    args = parser.parse_args()

    if args.dataset == 'searchsnippets':
        output_file = 'datasets/SearchSnippets/HuggingFace/'+ args.output_file
        file_path = 'datasets/SearchSnippets/SearchSnippets.txt'
        mat_path = 'datasets/SearchSnippets/SearchSnippets-STC2.mat'
        
    elif args.dataset == 'stackoverflow':
        file_path = 'datasets/stackoverflow/title_StackOverflow.txt'
        mat_path = 'datasets/stackoverflow/StackOverflow.mat'
        output_file = 'datasets/stackoverflow/HuggingFace/'+ args.output_file

    elif args.dataset == 'biomedical':
        file_path = 'datasets/Biomedical/Biomedical.txt'
        mat_path = 'datasets/Biomedical/Biomedical-STC2.mat'
        output_file = 'datasets/Biomedical/HuggingFace/'+ args.output_file

    else:
        raise ValueError("Invalid dataset")

    preprocessed_sentences = sentence_processor(file_path)
    list_sentence = sentence_encoder(args.model, preprocessed_sentences)

    # save embeddings to file
    with open(output_file, 'w') as file:
        for sentence in list_sentence:
            file.write(' '.join([str(i) for i in sentence]) + '\n')
    
    print("Done")
