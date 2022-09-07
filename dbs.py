from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from nltk.stem.snowball import SnowballStemmer
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from datasets import load_dataset
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from string import punctuation
from tqdm.notebook import tqdm
from pymystem3 import Mystem
import pandas as pd
import numpy as np
import openpyxl
import random
import torch
import nltk
import umap
import re

import string
import razdel

nltk.download("stopwords")
nltk.download('punkt')



class Preprocessor:
    def __init__(self, max_len=128, batch_size=32, model_name="bert-base-uncased"):
        self.batch_size = batch_size
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=max_len,
                                             return_tensors='pt')

    def prepare_data(self, text, label):
        text = list(map(self.tokenizer, text))
        label = list(map(int, label))
        num_clusters = len(set(label))
        label = torch.tensor(label)
        for i in range(len(text)):
            text[i]["input_ids"] = text[i]["input_ids"].squeeze(0)
            text[i]["token_type_ids"] = text[i]["token_type_ids"].squeeze(0)
            text[i]["attention_mask"] = text[i]["attention_mask"].squeeze(0)
            text[i]["label"] = label[i]
        shuffled_dataloader = DataLoader(text, batch_size=self.batch_size, shuffle=True)
        unshuffled_dataloader = DataLoader(text, batch_size=self.batch_size, shuffle=False)
        return shuffled_dataloader, unshuffled_dataloader, num_clusters

    def init_trec(self):
        dataset = load_dataset("trec")
        text = dataset["train"]["text"]
        label = dataset["train"]["label-coarse"]
        return text, label

    def init_dbpedia(self):
        dataset = load_dataset("dbpedia_14")
        text = dataset["train"]["content"]
        label = dataset["train"]["label"]
        random.seed(42)
        random.shuffle(text)
        random.seed(42)
        random.shuffle(label)
        text = text[:10000]
        label = label[:10000]
        return text, label

    def init_tweets(self):
        df = pd.read_csv('/home/jupyter/mnt/s3/shared-nlp-project-sirius/tweet_clean.csv')
        label = df['label'].tolist()
        text = df['text'].tolist()
        return text, label

    def init_news(self):
        df = pd.read_csv('/home/jupyter/mnt/s3/shared-nlp-project-sirius/TS_clean.csv')
        label = df['label'].tolist()
        text = df['text'].tolist()
        return text, label

    def init_short(self):
        with open("/home/jupyter/mnt/s3/shared-nlp-project-sirius/short-text-clustering-enchancment.txt", 'r') as f:
            data_ = list(map(lambda x: x.strip().split('\t'), f.readlines()))
            text = map(lambda x: x[1], data_)
            label = list(map(lambda x: int(x[0]), data_))
        return text, label

    def init_yelp(self):
        dataset = load_dataset("yelp_review_full")
        text = dataset["train"]["text"]
        label = dataset["train"]["label"]
        random.seed(42)
        random.shuffle(text)
        random.seed(42)
        random.shuffle(label)
        text = text[:10000]
        label = label[:10000]
        return text, label

class Metrics:
    def NMI(self, y, y_pred):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        return float(normalized_mutual_info_score(y, y_pred))

    def AR(self, y, y_pred):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        return float(adjusted_rand_score(y, y_pred))

    def cluster_accuracy(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

    def confusion_matrix_st(self, preds, labels):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

        num_inst = len(preds)
        num_labels = np.max(labels) + 1
        conf_matrix = np.zeros((num_labels, num_labels))

        for i in range(0, num_inst):
            gt_i = labels[i]
            pr_i = preds[i]
            conf_matrix[gt_i, pr_i] = conf_matrix[gt_i, pr_i] + 1

        return conf_matrix

class DatasetAutoEnc(Dataset):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def __getitem__(self, idx):
        return self.text['input_ids'][idx], self.text['attention_mask'][idx], self.text['token_type_ids'][idx]

    def __len__(self):
        return len(self.text['attention_mask'])

def batch_collate(batch):
    input_ids, attention_mask, label = torch.utils.data._utils.collate.default_collate(batch)
    max_length = attention_mask.sum(dim=1).max().item()
    attention_mask, input_ids = attention_mask[:, :max_length], input_ids[:, :max_length]
    return input_ids, attention_mask, label

class Plot_data():
    def umap_data(self, embeddings, n) -> np.array:
        ''' Dimension reduction function '''
        reducer = umap.UMAP(n_components=n)
        scaled_data = StandardScaler().fit_transform(embeddings)
        embedding = reducer.fit_transform(scaled_data)
        return embedding

    def centers(self, umap_data, labels):
        ln = len(np.unique(list(labels)))
        if -1 in np.unique(labels):
            ln -= 1
        centers = np.empty(shape=(len(np.unique(labels)) - 1, 4))
        for ind, class_ in enumerate(np.unique(labels)):
            x = 0
            y = 0
            i = 0
            for ind, elem in enumerate(umap_data):
                if labels[ind] == class_:
                    x += elem[0]
                    y += elem[1]
                    i += 1
            if class_ != -1 and class_ < len(centers): centers[class_] = [x / i, y / i,
                                                                          np.sqrt(i / len(labels) * 200000), class_]
        centers[:, 0] += abs(min(centers[:, 0]))
        centers[:, 1] += abs(min(centers[:, 1]))
        return centers

    def plt_data(emb, classes):
        umap_data = self.umap_data(emb, 2)
        arr = np.empty(shape=(len(umap_data, 3)))
        for ind, elem in enumerate(umap_data):
            arr[ind] = [elem, class_[ind]]
        return arr

def m_data(text, n):
    with tqdm(total=8) as tq:
        sk_corpus = main.preprocessing(text, False);
        tq.update(1)
        norm_data = main.to_texts(sk_corpus);
        tq.update(1)
        emb_data = main.model.encode(norm_data);
        tq.update(1)
        umap_data = main.umap_data(emb_data, n);
        tq.update(1)  # PCA(n_components=).fit_transform(emb_data)#self.umap_data(emb_data, 3); tq.update(1)
        token = [token for sublist in sk_corpus for token in sublist];
        tq.update(1)
        freq_tokens = Counter(token);
        tq.update(1)
        vectorizer = TfidfVectorizer(vocabulary=list(freq_tokens.keys()));
        tq.update(1)
        dbs = DBSCAN().fit(umap_data);
        tq.update(1)
        labels = dbs.labels_
    return labels

class Main:
    def __init__(self, data='/home/jupyter/mnt/s3/shared-nlp-project-sirius/vtb_dataset.xlsx', language='russian'):
        self.punctuation = punctuation + '»' + '«'
        self.lang = language
        self.download = True
        try:
            self.data = vtb = pd.read_excel(data, engine='openpyxl')['text'].tolist()[:5000]
        except Exception:
            self.download = False
        if language == 'russian':
            self.stem = Mystem()
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        else:
            self.stem = SnowballStemmer(language='english')
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = stopwords.words(language)

        if language == 'english':
            self.stop_words += ['i']
        else:
            self.stop_words += ['который', 'это', 'точно', 'вообще', 'просто']

    def mean_pooling(self, model_output, attention_mask) -> np.array:
        ''' Func for getting text embeddings '''
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_text_emb(self, data, tokenizer, model) -> np.array:
        ''' Func to get embeddigs with DeepPavlov model '''
        arr = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        encoded_input = DatasetAutoEnc(
            tokenizer(data, padding='max_length', truncation=True, max_length=128, return_tensors='pt'))
        loader = DataLoader(encoded_input, batch_size=100, shuffle=True)  # , collate_fn=batch_collate)
        with torch.no_grad():
            with tqdm(total=len(loader)) as tq:
                for i, batch in enumerate(loader):
                    input_, mask, type_ = batch
                    output = model(input_.to(device), mask.to(device), type_.to(device))
                    sentence_embeddings = self.mean_pooling(output, mask.to(device))
                    arr.append(sentence_embeddings.cpu().detach().numpy())
                    tq.update(1)
            arr = np.concatenate(arr)
        return arr

    def remove_urls(self, documents) -> list:
        return [re.sub('https?:\/\/.*?[\s+]', '', text) for text in documents]

    def replace_newline(self, documents):
        documents = [text.replace('\n', ' ') + ' ' for text in documents]
        return documents

    def remove_strange_symbols(self, documents):
        return [re.sub(f'[^A-Za-zА-Яа-яё0-9{string.punctuation}\ ]+', ' ', text) for text in documents]

    def tokenize(self, documents) -> list:
        if self.lang == 'english':
            return [nltk.word_tokenize(text) for text in documents]
        else:
            return [[token.text for token in razdel.tokenize(text)] for text in documents]

    def to_lower(self, documents) -> list:
        return [text.lower() for text in documents]

    def remove_punctuation(self, tokenized_documents):
        ttt = set(string.punctuation)
        return [[token for token in tokenized_text if not set(token) < ttt] for tokenized_text in tokenized_documents]

    def remove_numbers(self, documents):
        return [re.sub('(?!:\s)\d\.?\d*', ' ', text) for text in documents]

    def remove_stop_words(self, tokenized_documents) -> list:
        return [[token for token in tokenized_text if token not in self.stop_words] for tokenized_text in
                tokenized_documents]

    def lemmatize(self, documents) -> list:
        if self.lang == 'russian':
            return [''.join(self.stem.lemmatize(text)) for text in documents]
        else:
            return [' '.join(self.stem.stem(token) for token in text.split()) for text in documents]

    def join_text(self, clean_documents, cluster_ids):
        joined_texts = defaultdict(list)

        for clean_text, cluster_id in zip(clean_documents, cluster_ids):
            joined_texts[cluster_id].append(clean_text)

        for cluster_id, texts in joined_texts.items():
            joined_texts[cluster_id] = ' '.join(texts)
        return list(joined_texts.keys()), list(joined_texts.values())

    def preprocessing(self, documents):
        documents = self.replace_newline(documents)
        documents = self.remove_urls(documents)
        documents = self.remove_strange_symbols(documents)
        documents = self.to_lower(documents)
        documents = self.lemmatize(documents)
        documents = self.remove_numbers(documents)
        tokenized_documents = self.tokenize(documents)
        tokenized_documents = self.remove_stop_words(tokenized_documents)
        tokenized_documents = self.remove_punctuation(tokenized_documents)
        documents = [' '.join(tokenized_text) for tokenized_text in tokenized_documents]
        return documents

    def extract_keywords(self, cluster_names, vectorizer, tfidf_matrix, top_n):
        id_2_word = {token_id: word for word, token_id in vectorizer.vocabulary_.items()}
        ind_arr = [list(np.argsort(-x)) for x in tfidf_matrix]

        cluster_keywords = dict()

        for cluster_id, cluster_word_rating, cluster_word_tfidf in zip(cluster_names, ind_arr, tfidf_matrix):
            wl = []
            for word_id in cluster_word_rating[:top_n]:
                wl.append((id_2_word[word_id], cluster_word_tfidf[word_id]))
            cluster_keywords[cluster_id] = wl
        return cluster_keywords

    def get_tfifd_v2_keywords(self, documents, cluster_ids, top_n=20):
        clean_documents = self.preprocessing(documents)
        cluster_names, joined_texts = self.join_text(clean_documents, cluster_ids)

        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer.fit(clean_documents)
        idf = tf_idf_vectorizer.idf_
        vocab = tf_idf_vectorizer.vocabulary_

        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(joined_texts)

        # calculate tf-idf
        tf = X_counts / X_counts.sum(axis=1)
        tf_, idf_ = np.broadcast_arrays(tf, idf)
        X_tfidf = tf_ * idf_

        return self.extract_keywords(cluster_names, vectorizer, X_tfidf, top_n)

    def plot_data(self, class_, umap_data):
        arr = []
        for ind, elem in enumerate(umap_data):
            arr.append([elem, class_[ind]])
        return arr

    def get_json(self, tf_idf, data, umap_data, labels):

        plot_data_class = Plot_data()
        i_map = plot_data_class.centers(umap_data, labels)
        data = self.preprocessing(data)

        answer = {"status": "success", "payload": {"intertopic_map": [], "topics": [], "documents": []}}

        if not self.download:
            answer = {
                "status": "error",
                "project_id": 1
            }
            return answer

        for ind, dot in enumerate(i_map):
            answer["payload"]["intertopic_map"].append(
                {
                    "id": int(dot[3]),
                    "keywords": [x[0] for x in tf_idf[ind]][:5],
                    "size": dot[2],
                    "cord_x": dot[0],
                    "cord_y": dot[1]
                })
        
        for ind in range(len(labels)):
            answer["payload"]["documents"].append(
                {
                    "id": ind,
                    "cord_x": umap_data[ind][0],
                    "cord_y": umap_data[ind][1],
                    "cluster_id": labels[ind],
                    "description": [x for x in data[ind].split(" ") if x != " " and len(x) > 3][:7]
                }
            )
        return answer

    def predict(self):
        ''' Main function '''
        plot_data_class = Plot_data()
        emb_data = self.get_text_emb(self.data, self.tokenizer, self.model)
        umap_data = plot_data_class.umap_data(emb_data, 5)
        dbs = DBSCAN().fit(umap_data)
        labels = dbs.labels_
        tf_idf = self.get_tfifd_v2_keywords(self.data, labels)

        answer = self.get_json(tf_idf, self.data, umap_data, labels)

        return answer

main = Main()
info = main.predict()
print(info)
