# 📚 Course Recommendation System  
Content-Based · Collaborative Filtering (KNN / NMF / Neural) · Hybrid

> “Help learners find the next best course by analysing course descriptions, user ratings, and enrolment history.”

This repository contains a **full-stack recommender pipeline** that crawls course data, engineer features, trains multiple recommenders, evaluates their performance, and serves ranked course lists through a simple API / web interface.

---

## ✨ Key Components
| Layer | Script(s) | Description |
|-------|-----------|-------------|
| **Data collection** | `src/collection/data_collection.py` | Scrapes or ingests raw course-catalog & rating data (CSV / JSON). |
| **Exploratory analysis** | `src/eda/eda.py` | Generates summary stats & visualisations (plots are saved to `plots/`). |
| **Feature engineering** | `src/features/bow_features.py` | Builds TF-IDF / Bag-of-Words matrices from course titles & descriptions. |
| **Content-based models** | `src/content_based/` | • `content_based_similarity.py` (cosine-sim TF-IDF)<br>• `content_based_clustering.py` (K-means topic clusters)<br>• `content_based_user_profile.py` (user profile vectors). |
| **Collaborative filtering** | `src/collaborative/` | • `knn_collaborative_filtering.py`<br>• `nmf_collaborative_filtering.py`<br>• `nn_collaborative_filtering.py` (neural CF). |
| **Evaluation** | `src/collaborative/cf_evaluation.py` | Precision@K, Recall@K, NDCG, Coverage. |
| **Hybrid ranker** | `src/hybrid/course_similarity.py` | Combines content & CF scores with tunable weights. |
| **Serving layer** | `app.py` (FastAPI / Streamlit) | Exposes `GET /recommend?user_id=…&k=10` or interactive UI. |
| **Entry-point** | `main.py` | CLI glue: `python main.py train --model knn`, `... recommend`. |

---


## 📊 Results & Artifacts

- **Course-counts-per-genre bar chart** – 307 courses across 14 genres  
  *Largest:* **Database**, **Python** · *Smallest:* **FrontendDev**, **Blockchain**

- **User-enrollment distribution histogram** – most learners enrol in **1–5** courses; long-tail of power users

- **Top-20 most-popular courses** – horizontal bar chart

- **Word cloud of course titles** – dominant words: *Data · Science · Python · Machine*

---

### 📝 Content-Based Recommender
- User-profile-vs-genre flowchart  
- Course-similarity (BoW) flowchart **+ precision/recall table**  
- Delivers **≈ 2–4 unseen courses per user** at similarity ≥ 0.8

### 🗂️ Clustering-Based Recommender
- K-means flowchart  
- Delivers **≈ 3–5 unseen courses per user** (cluster model)

---

### 📐 Collaborative Filtering – RMSE

| Model | RMSE |
|-------|------|
| **KNN CF** | **0.750 158** |
| **NMF CF** | **0.569 733** ← *best* |
| Neural-CF | **2.890 297** |

*(bar chart comparison included in presentation)*

---

### 🔀 Hybrid Recommender
- Weighted blend: **0.4 × Content Similarity  +  0.6 × NMF CF**  
  *(architecture diagram in slides)*

---
### 🖼️ Demo UI
- Screenshot / GIF of the Streamlit app showing **top-10 course suggestions** for a sample user
- !(Path/Streamlit.png)

