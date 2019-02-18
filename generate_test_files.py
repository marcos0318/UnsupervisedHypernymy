import codecs
import numpy as np




dataset_path = "./datasets/"

files = ["BLESS.test", "EVALution.test", "LenciBenotto.test", "Weeds.test"]

emb_path = "/home/data/corpora/MultiplexEmbeddingMatrixAndEmbeddings/"

emb_files = ["2018_selected_dep.txt", "2018_selected_enwiki_glove_0.txt", "2018_selected_word2vec_0.txt"]

def load_dataset(dataset_file):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        
        # Only think hyper and True is positive
        # Not hyper nor True is negative

        filtered_dataset = []
        for t in dataset:
            x, y, label, relation = t
            if relation != "hyper" or label == "False":
                filtered_dataset.append([x[:-2], y[:-2], "False"])
            else:
                filtered_dataset.append([x[:-2], y[:-2], "True"])

        
    return filtered_dataset

def load_embeddings(embedding_file):
    with open(embedding_file, "r") as fin:
        embeddings = { line.strip().split()[0]: np.array(line.strip().split()[1:]) for line in fin.readlines()}
    return embeddings

# for f_name in files:
#     print(f_name, load_dataset(dataset_path + f_name))

for f_name in emb_files:
    print(f_name, load_embeddings(emb_path + emb_files))