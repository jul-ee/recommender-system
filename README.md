# 🎬 AutoInt+ 기반 영화 추천 시스템

본 프로젝트는 MovieLens 1M 데이터를 기반으로

1. EDA(탐색적 데이터 분석)를 통해 데이터 구조를 이해하고, AutoInt 모델을 활용해 사용자와 영화 간의 상호작용을 기반으로 평점을 예측합니다.
2. 딥러닝 모델 AutoInt+ 모델로 확장 후 하이퍼파라미터 조합에 대한 실험을 수행하고, 성능 비교를 통해 최적 모델을 도출합니다.

> 🛠 **Tech Stack** 
> : &nbsp;Python, TensorFlow, Pandas, NumPy, scikit-learn(전처리 및 평가), Streamlit(추천 결과 UI)

<br>
<br>

## README 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [데이터 설명](#데이터-설명)
3. [데이터 전처리](#데이터-전처리)
4. [모델 설계 및 학습-AutoInt](#모델-설계-및-학습-autoint)
5. [실험 진행 및 결과 기록](#실험-진행-및-결과-기록)
6. [추천 결과 시각화-Streamlit](#추천-결과-시각화-streamlit)
7. [인사이트 및 회고](#인사이트-및-회고)
8. [📂 디렉토리 구조](#-디렉토리-구조)

<br>
<br>

## 프로젝트 개요

- 분석 목표: AutoInt 기반 영화 추천 모델 구현 및 하이퍼파라미터 실험
- 데이터셋: &nbsp;[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/)
- 모델: &nbsp;AutoInt + MLP (AutoInt+ 구조)
- 평가지표: &nbsp;NDCG@10, Hit Ratio@10
- 시각화: &nbsp;Streamlit 앱을 통해 추천 결과 출력

<br>
<br>

## 데이터 설명

데이터는 다음 세 가지 파일로 구성됨

- `users.dat`: &nbsp;사용자 정보 (`user_id`, `gender`, `age`, `occupation`, `zip`)
- `ratings.dat`: &nbsp;평점 정보 (`user_id`, `movie_id`, `rating`, `timestamp`)
- `movies.dat`: &nbsp;영화 정보 (`movie_id`, `title`, `genres`)


<br>
<br>

## 데이터 전처리

- `ratings.dat`, `movies.dat`, `users.dat`를 CSV 형태로 변환 및 정제
- 파생 변수 생성:
    - 유저: 나이 그룹, 성별, 직업 → Label Encoding
    - 영화: 제목에서 연도/년대 추출 → `movie_year`, `movie_decade`
    - 장르: 장르 분리 추출 → `genre1`, ...
    - 평점: Unix timestamp 변환 → `rating_year`, ...
- 사용된 주요 feature
    
    ```
    ['user_id', 'gender', 'age', 'occupation', 'zipcode', 'movie_id',
     'genre1', 'genre2', 'genre3', 'release_year', 'release_decade',
     'rating_year', 'rating_month']
    
    ```
    
<br>
<br>

## 모델 설계 및 학습 (AutoInt+)

### ✓ &nbsp;모델 구성

- 모델 구조:
    - Embedding Layer
    - Multi-Head Self-Attention (3 layers, 2 heads, residual 포함)
    - MLP Layer (DNN 사용, BN 미사용, dropout=0.4)
    - Output: 이진 분류 (`rating >= 4` → positive)
- 기준 하이퍼파라미터:
    - Embedding dim = 16
    - Learning rate = 0.0001
    - Batch size = 2048
    - Epochs = 5

### ✓ &nbsp;모델 학습 및 저장

- Binary CrossEntropy + Adam Optimizer 사용
- 전체 사용자에 대해 평가 수행
- `model.save_weights()`로 가중치 저장 (Keras 호환 `.weights.h5` 포맷)


### ✓ &nbsp;모델 평가

>Top-10 추천 기준

- ndcg@10 ≒  **0.6617**
- Hit Ratio@10 ≒ **0.6302**


### ✓ &nbsp;최종 모델 저장 및 활용

- 저장 경로: `model/autoIntMLP_model_weights.weights.h5`
- 학습된 `LabelEncoder` 객체: `label_encoders.pkl`

```python
model.save_weights('model/autoIntMLP_model_weights.weights.h5')
```

    
<br>
<br>

## 실험 진행 및 결과 기록

> AutoInt+ 모델에 대해 다음과 같은 하이퍼파라미터 조합 실험 수행

```python
# 탐색한 파라미터 그리드
param_grid = {
    "embed_dim" : [8, 16],        # 임베딩 차원
    "dropout"   : [0.2, 0.4],     # 드롭아웃
    "lr"        : [1e-4],         # 학습률
    "batch"     : [1024, 2048],   # 배치사이즈
    "epochs"    : [3]             # epoch
}
```

#### 실험 결과

|  | **embed_dim** | **dropout** | **lr** | **batch** | **epochs** | **NDCG@10** | **HitRate@10** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.2 | 0.0001 | 1024 | 3 | 0.6619 | 0.6301 |
| 1 | 16 | 0.4 | 0.0001 | 1024 | 3 | 0.6617 | 0.6301 |
| 2 | 16 | 0.2 | 0.0001 | 2048 | 3 | 0.6615 | 0.6298 |
| 3 | 16 | 0.4 | 0.0001 | 2048 | 3 | 0.6609 | 0.6294 |
| 4 | 8 | 0.2 | 0.0001 | 1024 | 3 | 0.6602 | 0.6279 |
| 5 | 8 | 0.4 | 0.0001 | 1024 | 3 | 0.6598 | 0.6277 |
| 6 | 8 | 0.2 | 0.0001 | 2048 | 3 | 0.6592 | 0.6276 |
| 7 | 8 | 0.4 | 0.0001 | 2048 | 3 | 0.6589 | 0.6282 |

- 전반적으로 embed_dim=16, lr=0.0001 조합에서 가장 높은 성능을 보임
- 기존 실험 대비 NDCG@10과 Hit Ratio@10 값은 거의 동일한 수준(≈ 0.6617 / 0.6302) 을 유지하고 있어, 모델의 성능이 하이퍼파라미터 변화에 크게 민감하지 않음을 확인함

<br>
<br>

## 추천 결과 시각화 (Streamlit)

Streamlit 앱을 실행하면 선택한 유저 ID에 대해 추천 영화 목록을 시각적으로 확인할 수 있음

```bash
streamlit run show_st2.py
```


![02CF7E56-0EA9-477B-BD65-E8D04CC96C54_1_201_a](https://github.com/user-attachments/assets/35c71872-42ca-4934-9d3c-98bccc50d49f)


- 사용자 정보, 기존 시청 목록, 추천 영화 리스트 출력
- AutoInt+ 모델의 예측 확률 기준으로 상위 N개 영화 추천

<br>
<br>


## 인사이트 및 회고

딥러닝 기반 AutoInt+ 모델의 설계와 실험 자동화를 통해 추천 시스템 전반을 간단히 설계하고 이해할 수 있었음.

범주형 피처 간의 복잡한 상호작용을 attention으로 효과적으로 모델링하고, 전통적인 협업 필터링보다 더 유연한 구조임일 수 있음을 확인함.

실험 기반 접근을 통해 하이퍼파라미터 변화에 따른 성능 유지 혹은 미세한 변화를 체계적으로 관찰했으며 향후 더 복잡한 모델 비교나 도메인 확장으로 적용 가능함.


<br>
<br>

## 📂 디렉토리 구조

```bash
📂 recommender_system_autoint_project/
├── data/
│   └── ml-1m/
│       ├── ratings_prepro.csv  # 전처리된 평점 데이터
│       ├── movies_prepro.csv   # 전처리된 영화 데이터
│       └── users_prepro.csv    # 전처리된 사용자 데이터
│
├── model/
│   └── autoIntMLP_model_weights.weights.h5  # 모델 가중치 저장
│
├── recommender_system/              # 추천 시스템 실습 프로젝트
│   ├── 01_artist_recommender.ipynb  # 유사한 아티스트 추천 시스템
│   ├── 02_item_recommender.ipynb    # 다음에 구매할 아이템 예측 추천 시스템
│   └── 03_movie_recommender.ipynb   # 영화 SBR 추천 시스템
│
├── autoint.py                 # AutoInt 모델 정의
├── autointmlp.py              # AutoInt+ 모델 정의
│
├── show_st.py                 # AutoInt 모델 기반 Streamlit 앱
├── show_st2.py                # AutoInt+ 모델 기반 Streamlit 앱
│
├── pre_project.ipynb          # 데이터 분석 및 전처리 노트북
└── label_encoders.pkl         # LabelEncoder 객체 저장

```

<br>

> [참고]  
> AutoInt 논문: [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)

> 본 프로젝트는 학습 및 실험 목적으로 진행되었으며, MovieLens 데이터셋의 라이선스를 준수합니다.
