# Báo cáo

1. Chuẩn bị dữ liệu

Từ viwiki tải file [viwiki-20240701-pages-articles-multistream4.xml-p4565247p6065246](/workspaces/NLP-ASM2/viwiki-20240701-pages-articles-multistream4.xml-p4565247p6065246), sử dụng wikiextractor lấy được nội dung [wiki_00](/workspaces/NLP-ASM2/AA/wiki_00)

2. Tiền xử lý dữ liệu

```Python
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents

def word_segment(sent):
    sent = tokenize(sent)
    return sent
```

Kết quả ta có file [datatrain.txt](/workspaces/NLP-ASM2/datatrain.txt)

![datatrain.txt](/workspaces/NLP-ASM2/image/datatrain.png)


3. Train mô hình sử dụng word2vec và fastText

```python
from gensim.models import Word2Vec
import os

# Đường dẫn tới thư mục lưu trữ mô hình
model_dir = '/content/drive/MyDrive/Colab_Notebooks/NLP/Week2/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Đường dẫn tới dữ liệu huấn luyện
path_data = '/content/drive/MyDrive/Colab_Notebooks/NLP/Week2/datatrain.txt'

def read_data(path):
    train_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(line.split())
    return train_data

if __name__ == '__main__':
    # Đọc dữ liệu
    train_data = read_data(path_data)
    
    # Huấn luyện mô hình Word2Vec
    model_w2v = Word2Vec(sentences=train_data, vector_size=150, window=10, min_count=2, workers=4, sg=0)
    
    # Lưu mô hình Word2Vec đầy đủ
    model_file = os.path.join(model_dir, "word2vec.model")
    try:
        model_w2v.save(model_file)
        print(f"Model saved successfully to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

```

```python
from gensim.models import FastText
import os

# Đường dẫn tới thư mục lưu trữ mô hình
model_dir = '/content/drive/MyDrive/Colab_Notebooks/NLP/Week2/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Đường dẫn tới dữ liệu huấn luyện
path_data = '/content/drive/MyDrive/Colab_Notebooks/NLP/Week2/datatrain.txt'

def read_data(path):
    train_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(line.split())
    return train_data

if __name__ == '__main__':
    # Đọc dữ liệu
    train_data = read_data(path_data)
    
    # Huấn luyện mô hình fastText
    model_fasttext = FastText(vector_size=150, window=10, min_count=2, workers=4, sg=1)
    model_fasttext.build_vocab(corpus_file=path_data)
    model_fasttext.train(corpus_file=path_data, total_examples=model_fasttext.corpus_count, total_words=model_fasttext.corpus_total_words, epochs=10)
    
    # Lưu mô hình
    model_file = os.path.join(model_dir, "fasttext.model")
    try:
        model_fasttext.save(model_file)
        print(f"Model saved successfully to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")
```

4. Demo

![demo](/workspaces/NLP-ASM2/image/demo.png)
