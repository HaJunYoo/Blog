---
title: Neural Collaborative Filtering
description: Neural Collaborative Filtering 실습
toc: true 
badges: true
comments: false
categories: [datascience, project]
image:
---

# Neural Collaborative Filtering 개요

본 실습은 패스트캠퍼스의 `'딥러닝을 활용한 추천시스템 구현 올인원 패키지 Online'` 을 듣고 작성하였다는 점을 명시합니다.

- user과 item의 latent features를 모델링하기 위한 신경망 구조를 제안한다 ⇒ user과 item의 관계를 보다 복잡하게 모델링할 수 있다는 점.
- Multi layer Perceptron을 사용
- Neural net 기반의 Collaborative filtering으로 non linear한 부분을 커버했다. ⇒ 기존의 Linear Matrix Factorization의 한계점을 지적하였다.

---


1. [논문](https://arxiv.org/pdf/1708.05031.pdf)
2. Keras로 작성된 [저자 코드](https://github.com/hexiangnan/neural_collaborative_filtering)
3. 논문은 0과 1로 user-item interaction으로 matrix을 나타내고 학습했으나, 이번 실습에서는 rating을 직접 예측하고, loss를 구해보는 것을 진행한다

## Configuration


```python
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import math
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
```

## Load Dataset
- KMRD 데이터셋 활용
- google colab의 경우 data path 다시 확인하기


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
data_path = '/content/drive/MyDrive/추천 시스템/fastcampus-RecSys/data/kmrd/kmr_dataset/datafile/kmrd-small'
```


```python
def read_data(data_path):
  df = pd.read_csv(os.path.join(data_path,'rates.csv'))
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=1234, shuffle=True)
  return train_df, val_df
```


```python
!nvidia-smi
```

    Fri Jul  1 16:21:37 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   37C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+



```python
# 학습할 영화 데이터 분석
train_df, val_df = read_data(data_path)
```


```python
print(train_df.shape)
print(train_df.head())
```

    (112568, 4)
             user  movie  rate        time
    137023  48423  10764    10  1212241560
    92868   17307  10170    10  1122185220
    94390   18180  10048    10  1573403460
    22289    1498  10001     9  1432684500
    80155   12541  10022    10  1370458140



```python
val_df.shape
```




    (28142, 4)




```python
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(12,7))
ax = ax.ravel()

train_df['rate'].hist(ax=ax[0])
val_df['rate'].hist(ax=ax[1])
```



![output]({{site.baseurl}}/images/recommends/output_12_1.png)    


대부분 `10`점에 수렴한 상태이다. 



```python
train_df['rate'].describe()
```




    count    112568.000000
    mean          8.948369
    std           2.114602
    min           1.000000
    25%           9.000000
    50%          10.000000
    75%          10.000000
    max          10.000000
    Name: rate, dtype: float64



## Load movie dataframe


```python
# Load all related dataframe
movies_df = pd.read_csv(os.path.join(data_path, 'movies.txt'), sep='\t', encoding='utf-8')
movies_df = movies_df.set_index('movie')

castings_df = pd.read_csv(os.path.join(data_path, 'castings.csv'), encoding='utf-8')
countries_df = pd.read_csv(os.path.join(data_path, 'countries.csv'), encoding='utf-8')
genres_df = pd.read_csv(os.path.join(data_path, 'genres.csv'), encoding='utf-8')

# Get genre information
genres = [(list(set(x['movie'].values))[0], '/'.join(x['genre'].values)) for index, x in genres_df.groupby('movie')]
combined_genres_df = pd.DataFrame(data=genres, columns=['movie', 'genres'])
combined_genres_df = combined_genres_df.set_index('movie')

# Get castings information
castings = [(list(set(x['movie'].values))[0], x['people'].values) for index, x in castings_df.groupby('movie')]
combined_castings_df = pd.DataFrame(data=castings, columns=['movie','people'])
combined_castings_df = combined_castings_df.set_index('movie')

# Get countries for movie information
countries = [(list(set(x['movie'].values))[0], ','.join(x['country'].values)) for index, x in countries_df.groupby('movie')]
combined_countries_df = pd.DataFrame(data=countries, columns=['movie', 'country'])
combined_countries_df = combined_countries_df.set_index('movie')

movies_df = pd.concat([movies_df, combined_genres_df, combined_castings_df, combined_countries_df], axis=1)

```


```python
movies_df.head()
```





  <div id="df-c04c21ee-9624-47e1-9d58-b913c84546d0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>title_eng</th>
      <th>year</th>
      <th>grade</th>
      <th>genres</th>
      <th>people</th>
      <th>country</th>
    </tr>
    <tr>
      <th>movie</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10001</th>
      <td>시네마 천국</td>
      <td>Cinema Paradiso , 1988</td>
      <td>2013.0</td>
      <td>전체 관람가</td>
      <td>드라마/멜로/로맨스</td>
      <td>[4374, 178, 3241, 47952, 47953, 19538, 18991, ...</td>
      <td>이탈리아,프랑스</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>빽 투 더 퓨쳐</td>
      <td>Back To The Future , 1985</td>
      <td>2015.0</td>
      <td>12세 관람가</td>
      <td>SF/코미디</td>
      <td>[1076, 4603, 917, 8637, 5104, 9986, 7470, 9987]</td>
      <td>미국</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>빽 투 더 퓨쳐 2</td>
      <td>Back To The Future Part 2 , 1989</td>
      <td>2015.0</td>
      <td>12세 관람가</td>
      <td>SF/코미디</td>
      <td>[1076, 4603, 917, 5104, 391, 5106, 5105, 5107,...</td>
      <td>미국</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>빽 투 더 퓨쳐 3</td>
      <td>Back To The Future Part III , 1990</td>
      <td>1990.0</td>
      <td>전체 관람가</td>
      <td>서부/SF/판타지/코미디</td>
      <td>[1076, 4603, 1031, 5104, 10001, 5984, 10002, 1...</td>
      <td>미국</td>
    </tr>
    <tr>
      <th>10005</th>
      <td>스타워즈 에피소드 4 - 새로운 희망</td>
      <td>Star Wars , 1977</td>
      <td>1997.0</td>
      <td>PG</td>
      <td>판타지/모험/SF/액션</td>
      <td>[1007, 535, 215, 1236, 35]</td>
      <td>미국</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c04c21ee-9624-47e1-9d58-b913c84546d0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c04c21ee-9624-47e1-9d58-b913c84546d0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c04c21ee-9624-47e1-9d58-b913c84546d0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 논문:  user latent vector + item latent vector
- 새롭게 생각할 수 있는 방법: user latent vector + item latent vector + etc vector (예시) meta information 

- 논문의 아이디어를 가져와서 `내 상황에 맞게끔 적용`하는 것이 추천시스템의 본질이다.


```python
# 영화 데이터의 메타 정보를 확인한다
movieName_dict = movies_df.to_dict()['title']
genres_dict = movies_df.to_dict()['genres']
```


```python
movies_df['genres']
```




    movie
    10001       드라마/멜로/로맨스
    10002           SF/코미디
    10003           SF/코미디
    10004    서부/SF/판타지/코미디
    10005     판타지/모험/SF/액션
                 ...      
    10995              스릴러
    10996              코미디
    10997               공포
    10998    드라마/액션/모험/스릴러
    10999        SF/드라마/공포
    Name: genres, Length: 999, dtype: object



## Dataset Loader 


```python
class DatasetLoader:
    def __init__(self, data_path):
        self.train_df, val_temp_df = read_data(data_path) # 데이터를 읽어와서

        self.min_rating = min(self.train_df.rate) # 최소 평점
        self.max_rating = self.train_df.rate.max() # 최대 평점 

        self.unique_users = self.train_df.user.unique() # 유니크한 유저 얼마나 되는지 확인 
        self.num_users = len(self.unique_users)
        self.user_to_index = {original: idx for idx, original in enumerate(self.unique_users)}
        # 인덱스로 바꿔줌 => 어느 인덱스가 1이되는지 원핫벡터로 : 0 1 0 0 0 ... 0 

        self.unique_movies = self.train_df.movie.unique()
        self.num_movies = len(self.unique_movies)
        self.movie_to_index = {original: idx for idx, original in enumerate(self.unique_movies)}

        self.val_df = val_temp_df[val_temp_df.user.isin(self.unique_users) & val_temp_df.movie.isin(self.unique_movies)]

    def generate_trainset(self):
        # user 0, 0, 0, 1,2, 3,3, -> movie: 0,0,0,0,0,0,
        X_train = pd.DataFrame({'user': self.train_df.user.map(self.user_to_index),
                     'movie': self.train_df.movie.map(self.movie_to_index)})
        y_train = self.train_df['rate'].astype(np.float32)

        return X_train, y_train

    def generate_valset(self):
        X_val = pd.DataFrame({'user': self.val_df.user.map(self.user_to_index),
                              'movie': self.val_df.movie.map(self.movie_to_index)})
        y_val = self.val_df['rate'].astype(np.float32)
        return X_val, y_val

```

## Model Structure
- 논문에서 제시한 모델 구조를 그대로 구현하고 영화 데이터로 실습해본다. 
- User Vector는 전체 영화 데이터에서 영화를 평가한 userid를 onehot vector로 나타낸 형태
- Item Vector는 전체 영화 데이터에 등장하는 영화의 id를 onehot vector로 나타낸 형태




```python
class FeedForwardEmbedNN(nn.Module):

    def __init__(self, n_users, n_movies, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors) # number of user만큼 사이즈를 만들어서 원 핫으로 임베딩해야한다 
        self.movie_emb = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*2, hidden, dropouts)))
        self.fc = nn.Linear(hidden[-1], 1)

    # hidden(은닉층) 개수만큼 리니어(계층) 세트를 만들어내서 계속 붙여주는 작업 => 총 3개가 생김
    def generate_layers(self, n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)

        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield nn.Linear(n_factors, hidden[idx])
            else:
                yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropouts[idx])

            idx += 1

    def forward(self, users, movies, min_rating=0.5, max_rating=5):
        concat_features = torch.cat([self.user_emb(users), self.movie_emb(movies)], dim=1) 
        # 유저와 유저 임베딩을 가지고 concat을 함
        x = F.relu(self.hidden_layers(concat_features)) # relu를 씌우고 
        # 0과 1사이의 숫자로 나타낸다
        out = torch.sigmoid(self.fc(x))
        # rating으로 변환한다
        out = (out * (max_rating - min_rating)) + min_rating # 0~1의 값은 변환을 해서 출력

        return out

    def predict(self, users, movies): # predict score을 내보냄 
        # return the score
        output_scores = self.forward(users, movies)
        return output_scores
```


```python
class BatchIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]

```


```python
def batches(X, y, bs=32, shuffle=True): # 배치사이즈 32
    for x_batch, y_batch in BatchIterator(X, y, bs, shuffle):
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)
        yield x_batch, y_batch.view(-1, 1)
```

## Train model
데이터셋과 모델 학습에 필요한 configuration을 입력하고, 학습을 하는 함수를 만든다
configuration을 바꾸면서 모델의 성능을 측정해볼 수 있다. 


```python
def model_train(ds, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = ds.generate_trainset()
    X_valid, y_valid = ds.generate_valset()
    print(f'TrainSet Info: {ds.num_users} users, {ds.num_movies} movies')

    model = FeedForwardEmbedNN(
        n_users=ds.num_users, n_movies=ds.num_movies,
        n_factors=config['num_factors'], hidden=config['hidden_layers'],
        embedding_dropout=config['embedding_dropout'], dropouts=config['dropouts']
    ) # 모델을 정의 
    model.to(device)

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    max_patience = config['total_patience']
    num_patience = 0
    best_loss = np.inf

    criterion = nn.MSELoss(reduction='sum') # MSE LOSS를 사용  <- 논문은 cross entropy를 사용
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # 일반적으로 많이 사용하는 아담 옵티마이저 사용

    result = dict()
    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        for batch in batches(X_train, y_train, shuffle=True, bs=batch_size):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            # with torch.no_grad() 와 동일한 syntax 입니다
            with torch.set_grad_enabled(True):
                outputs = model(x_batch[:, 0], x_batch[:, 1], ds.min_rating, ds.max_rating)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            training_loss += loss.item() # 로스 값을 계속 더해줌 
        result['train'] = training_loss / len(X_train) #로스의 평균 값을 해줌

        # Apply Early Stopping criteria and save best model params
        val_outputs = model(torch.LongTensor(X_valid.user.values).to(device),
                            torch.LongTensor(X_valid.movie.values).to(device), ds.min_rating, ds.max_rating)
        val_loss = criterion(val_outputs.to(device), torch.FloatTensor(y_valid.values).view(-1, 1).to(device))
        result['val'] = float((val_loss / len(X_valid)).data)

        if val_loss < best_loss:
            print('Save new model on epoch: %d' % (epoch + 1))
            best_loss = val_loss
            result['best_loss'] = val_loss
            torch.save(model.state_dict(), config['save_path'])
            num_patience = 0
        else:
            num_patience += 1

        print(f'[epoch: {epoch+1}] train: {result["train"]} - val: {result["val"]}')

        if num_patience >= max_patience:
            print(f"Early Stopped after epoch {epoch+1}")
            break

    return result
```


```python
# model valid는 학습한 모델을 로드함 => 저장된 모델을 가지고 predict를 수행 
def model_valid(user_id_list, movie_id_list, data_path):
    dataset = DatasetLoader(data_path)
    processed_test_input_df = pd.DataFrame({
        'user_id': [dataset.user_to_index[x] for x in user_id_list],
        'movie_id': [dataset.movie_to_index[x] for x in movie_id_list]
    })

    # 학습한 모델 load하기 
    my_model = FeedForwardEmbedNN(dataset.num_users, dataset.num_movies,
                       config['hidden_layers'], config['dropouts'], config['num_factors'], config['embedding_dropout'])
    my_model.load_state_dict(torch.load('params.data')) # 모델을 로드
    prediction_outputs = my_model.predict(users=torch.LongTensor(processed_test_input_df.user_id.values),
                     movies=torch.LongTensor(processed_test_input_df.movie_id.values)) # 모델로 예측을 수행 

    return prediction_outputs
```


```python
dataset = DatasetLoader(data_path) # 데이터 셋을 로드
```


```python
config = {
  "num_factors": 16,
  "hidden_layers": [64, 32, 16],
  "embedding_dropout": 0.05,
  "dropouts": [0.3, 0.3, 0.3],
  "learning_rate": 1e-3,
  "weight_decay": 1e-5,
  "batch_size": 8,
  "num_epochs": 3,
  "total_patience": 30,
  "save_path": "params.data"
} 
# configuration 정의
```


```python
model_train(dataset, config) # epoch 3
```

    TrainSet Info: 44453 users, 597 movies


     33%|███▎      | 1/3 [00:39<01:19, 39.75s/it]

    Save new model on epoch: 1
    [epoch: 1] train: 4.342063758400752 - val: 3.8571109771728516


     67%|██████▋   | 2/3 [01:13<00:36, 36.42s/it]

    Save new model on epoch: 2
    [epoch: 2] train: 3.7550543531024645 - val: 3.582547426223755


    100%|██████████| 3/3 [01:48<00:00, 36.10s/it]

    Save new model on epoch: 3
    [epoch: 3] train: 3.3175223239590426 - val: 3.543619394302368


    





    {'best_loss': tensor(71779.5547, device='cuda:0', grad_fn=<MseLossBackward0>),
     'train': 3.3175223239590426,
     'val': 3.543619394302368}




```python
val_df.head()
```





  <div id="df-5c60993d-0171-47d4-9249-28bce80eeca2">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>movie</th>
      <th>rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76196</th>
      <td>11242</td>
      <td>10253</td>
      <td>10</td>
      <td>1437788760</td>
    </tr>
    <tr>
      <th>109800</th>
      <td>26903</td>
      <td>10102</td>
      <td>10</td>
      <td>1322643900</td>
    </tr>
    <tr>
      <th>60479</th>
      <td>7101</td>
      <td>10007</td>
      <td>1</td>
      <td>1314804000</td>
    </tr>
    <tr>
      <th>71460</th>
      <td>9705</td>
      <td>10016</td>
      <td>10</td>
      <td>1228825200</td>
    </tr>
    <tr>
      <th>73864</th>
      <td>10616</td>
      <td>10106</td>
      <td>8</td>
      <td>1425046200</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5c60993d-0171-47d4-9249-28bce80eeca2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5c60993d-0171-47d4-9249-28bce80eeca2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5c60993d-0171-47d4-9249-28bce80eeca2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
movie_id_list = [10253, 10102, 10007]
user_id = 11242
user_id_list = [user_id] * len(movie_id_list)
pred_results = [float(x) for x in model_valid(user_id_list, movie_id_list, data_path)] # 저장된 모델로 예측을 함 

result_df = pd.DataFrame({
    'userId': user_id_list,
    'movieId': movie_id_list,
    # 'movieName': [movieName_dict[x] for x in movie_id_list],
    # 'genres': [genres_dict[x] for x in movie_id_list],
    'pred_ratings': pred_results
})

result_df.sort_values(by='pred_ratings', ascending=False)
```





  <div id="df-39128870-72c8-4a2b-90b9-dc6d62611eed">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>pred_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11242</td>
      <td>10253</td>
      <td>4.825642</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11242</td>
      <td>10102</td>
      <td>4.768200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11242</td>
      <td>10007</td>
      <td>4.103816</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-39128870-72c8-4a2b-90b9-dc6d62611eed')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-39128870-72c8-4a2b-90b9-dc6d62611eed button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-39128870-72c8-4a2b-90b9-dc6d62611eed');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python

```
