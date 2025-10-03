# Philly Restaurants — Unsupervised Communities & Cold‑Start Recommender

## The Cold-Start Problem

Traditional restaurant recommendation systems face a critical challenge: **new restaurants** with minimal reviews cannot be effectively recommended using collaborative filtering (no user-item interaction history), and **new users** with no history receive poor recommendations. This creates barriers to discovery for new businesses and limits user exploration.

**Our Solution**: Use **unsupervised learning** to discover natural restaurant groupings based on intrinsic characteristics (review text, cuisine, price, location, quality) rather than interaction matrices. By extracting latent topics from reviews and clustering restaurants in a fused feature space, we enable effective **zero-history recommendations** for both new restaurants and new users.

## Key Results

- **K-Means clustering**: Achieved silhouette score of **0.241** with **perfect stability (ARI=1.0)**, discovering **4 interpretable clusters** suitable for business reporting
- **Content-based recommender**: Achieved **2.21% Recall@10** (9.2× better than random baseline) in cold-start scenarios with zero interaction history
- **Topic modeling**: Extracted 10 interpretable topics (pizza, sushi, coffee shops, bars, sandwiches, etc.) enabling preference-based recommendations
- **Model comparison**: Evaluated K-Means, GMM, and Agglomerative clustering; K-Means selected for optimal stability-quality balance

---

## Contents

- Notebook: `philly_restaurants_streamlined.ipynb` (end‑to‑end EDA → modeling → recommender → evaluation)
- Data folders: `Yelp JSON/yelp_dataset/` (businesses/reviews) and `data/` (optional local mirror)
- Outputs: Generated inline within the notebook (figures, tables, evaluation metrics)

Repository structure (trimmed):

```
unsupervised/
├─ philly_restaurants_streamlined.ipynb
├─ README.md
└─ Yelp JSON/
   └─ yelp_dataset/
      ├─ yelp_academic_dataset_business.json
      └─ yelp_academic_dataset_review.json
```

---

## Dataset & Provenance

**Source**: [Yelp Open Dataset](https://www.yelp.com/dataset) (provided for academic research and educational purposes)

**Data Collection Method**: Yelp's platform where users voluntarily submit reviews and ratings; Yelp releases periodic snapshots in JSON format

**Dataset Composition**:
- ~150K businesses and ~7M reviews across multiple cities (full dataset)
- Philadelphia filter: 10,548 businesses → 4,136 eateries after filtering
- Review sample: Reservoir sampling of 500K reviews → 39,061 reviews for Philadelphia eateries
- Temporal coverage: Reviews span multiple years through 2022

**Filtering Strategy**:
- **Geographic**: City == "Philadelphia" AND is_open == 1
- **Category inclusion**: Explicit eatery anchors (restaurants, coffee & tea, pizza, bakeries, food trucks, sandwiches, etc.) totaling 60+ food-related categories
- **Category exclusion**: Explicit non-eatery removal (beauty & spas, hair salons, barbers, nail salons, tattoo, massage, etc.)
- **Result**: 4,136 true eateries (excluded 6,412 non-eatery businesses)

**Path Resolution**: Notebook checks `YELP_DATA_DIR` env var → `./Yelp JSON/yelp_dataset` → `./data/yelp` (first valid path wins)

**Ethical Considerations**: Data anonymized by Yelp (pseudonymized user IDs); used strictly for academic/educational purposes under Yelp's dataset license

**Known Biases**:
- Tourist/visitor skew (more likely to review than locals)
- Extreme experience bias (very good/very bad more likely to be reviewed)
- Temporal drift (pre/post-pandemic dining patterns)
- Self-selection bias (Yelp user demographics not representative of full population)

---

## Methodology Overview

### 1. Exploratory Data Analysis (EDA)
**Purpose**: Understand data quality, distributions, and relationships to inform modeling decisions

- **Data loading & filtering**: Extract Philadelphia restaurants from Yelp Open Dataset; apply multi-stage category filtering (include eateries, exclude beauty/salon/spa businesses)
- **Feature engineering**:
  - Sentiment analysis via VADER (mean & std per restaurant)
  - Recency weighting (exponential decay, 1-year half-life)
  - Geographic distance from city center (Haversine formula)
  - Price level imputation (median fill for 20% missing)
- **Text processing**: Aggregate reviews per restaurant → TF-IDF (1-2 grams, 4000 features) → NMF (10 topics) for interpretation + TruncatedSVD (100 dims) for clustering
- **Visualizations**: Distributions, correlations, boxplots (outliers), geographic scatter, missingness heatmap
- **Key findings**:
  - Review counts and text length heavily right-skewed (log-transformed)
  - 80% of restaurants in $ or $$ price range
  - Weak correlations between most features (multimodality beneficial)
  - ~18% missing review-derived features (imputed with medians/defaults)

### 2. Feature Fusion & Hyperparameter Tuning
**Purpose**: Combine text semantics with numeric/categorical features for optimal clustering

- **Fusion approach**: Concatenate weighted text embeddings (SVD) + standardized numeric features
- **Grid search over**:
  - Fusion weights: w_text ∈ [1.0], w_num ∈ [0.3, 0.7]
  - Number of clusters K ∈ [3, 8]
  - UMAP: n_neighbors ∈ [15, 30, 50], min_dist=0.1
  - HDBSCAN: min_cluster_size ∈ [15, 20, 30], min_samples ∈ [5, 10]
  - GMM: covariance_type ∈ ['full', 'tied', 'diag', 'spherical']
- **Optimal configuration**: w_text=1.0, w_num=0.7, K=4 (K-Means), UMAP n_neighbors=50, HDBSCAN min_cluster_size=20/min_samples=5

### 3. Unsupervised Clustering Models Compared
**Purpose**: Identify best approach for discovering restaurant groupings

| Model | Silhouette | CH Index | DB Index | Clusters | Stability (ARI) |
|-------|-----------|----------|----------|----------|----------------|
| **K-Means** | **0.241** | **948** | **1.322** | 4 | **1.00** |
| GMM | 0.232 | 825 | 1.269 | 4 | — |
| Agglomerative | 0.187 | 794 | 1.521 | 4 | — |

**Why K-Means wins**:
- Perfect stability (ARI=1.00) ensures reproducible business insights
- Best overall silhouette score among traditional methods
- Simple interpretation for stakeholder communication
- Hyperparameter tuning identified K=4 as optimal across range [3,8]

**Trade-off**: K-Means sacrifices some fine-grained separation for perfect reproducibility and interpretability

### 4. Cluster Interpretation
**Purpose**: Understand the discovered restaurant groupings

**K-Means 4-Cluster Solution**:
- **Cluster 0** (n=2,219, 54%): Mainstream mid-to-high quality casual dining (4.1 stars, $$, topic_6 dominant)
- **Cluster 2** (n=1,259, 31%): Lower-rated or sparse-review establishments (2.3 stars, $$, topic_0 dominant)
- **Cluster 3** (n=551, 13%): Suburban/peripheral neighborhood spots (3.7 stars, $$, topic_0, 12.5km from center)
- **Cluster 1** (n=70, 2%): High-activity popular hotspots (4.1 stars, $$, topic_6, 99 median reviews)

**Topic dominance**: Review content (topics) is primary clustering driver, with quality (stars) and location (km_from_center) providing stratification

### 5. Cold-Start Recommender System
**Purpose**: Enable recommendations without user-item interaction history

**Algorithm**:
```
score = 0.6 × content_similarity + 0.25 × quality_prior + 0.15 × geo_price_affinity
```

**Components**:
- **Content similarity**: Cosine distance in fused feature space (item-item) OR user topic preference vector (user cold-start)
- **Quality prior**: 60% stars + 30% sentiment + 10% recency
- **Affinity filters**: Exponential decay penalties for out-of-range price/distance

**Cold-start scenarios supported**:
1. **New restaurant (0-10 reviews)**: Assign to cluster based on available content → recommend to users preferring that cluster
2. **New user (zero history)**: User inputs topic preferences (e.g., "I like coffee shops and sandwiches") + optional price/distance filters → content-based matching
3. **Exploration mode**: Community signals surface cross-cuisine recommendations

**Performance**:
- **Recall@10**: 2.21% (hit rate = 1 in 45 users get relevant recommendation in top-10)
- **vs. Random baseline**: 9.2× better (random = 0.24%)
- **Diversity**: Mean cluster entropy 0.26 (recommendations span ~1.3 clusters, focused but not monolithic)
- **Users evaluated**: 271 users with ≥8 reviews (time-aware 80/20 train/test split)

### 6. Evaluation & Validation
**Clustering metrics**: Silhouette (cohesion/separation), Calinski-Harabasz (density), Davies-Bouldin (cluster similarity), stability (ARI/NMI across random seeds)

**Recommender evaluation**: Time-aware split (80/20 train/test per user with ≥8 reviews), Recall@K (hit rate), cluster entropy (diversity)

**Qualitative validation**: Manual inspection of cold-start scenarios confirms topic-driven recommendations match user intent

---

## How to run

1) Open the notebook `philly_restaurants_streamlined.ipynb` and run cells top‑to‑bottom.
2) Ensure Yelp JSON files are available; optionally set `YELP_DATA_DIR` to point to the dataset directory.
3) Dependencies are standard scientific Python stack (pandas, numpy, scikit‑learn, matplotlib, seaborn, nltk). The notebook will download NLTK VADER lexicon automatically if missing.

Optional environment setup (if running outside VS Code/Jupyter with managed kernels):

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install -U pandas numpy scikit-learn matplotlib seaborn nltk
```

---

## Results & Key Insights

### Clustering Results

**K-Means discovered 4 interpretable clusters**:
- **Cluster 0** (n=2,219, 54%): Mainstream mid-to-high quality casual dining (4.1 stars, $$, topic_6 dominant)
- **Cluster 2** (n=1,259, 31%): Lower-rated or sparse-review establishments (2.3 stars, $$, topic_0 dominant)
- **Cluster 3** (n=551, 13%): Suburban/peripheral neighborhood spots (3.7 stars, $$, 12.5km from center)
- **Cluster 1** (n=70, 2%): High-activity popular hotspots (4.1 stars, $$, 99 median reviews)

**Topic Distribution**:
- **Topic 6** (full-service restaurants): Dominant in high-quality clusters 0 and 1
- **Topic 0** (general service/ordering): Dominant in lower-rated cluster 2 and suburban cluster 3
- Other topics (pizza, coffee, bars, sushi, sandwiches, tacos, desserts, Asian) distribute across clusters

### Cold-Start Recommender Performance

**Quantitative**:
- 2.21% Recall@10 with zero interaction history (9.2× random baseline)
- Focused diversity (cluster entropy 0.26 = ~1.3 clusters per recommendation set)
- 271 users evaluated with time-split validation (80/20 train/test)

**Qualitative validation**:
- **Scenario 1** (budget + nightlife preferences): ✅ Correctly returned affordable bars, pizza, casual spots within 3km
- **Scenario 2** (cheesesteak preferences): ✅ Perfectly matched topic 5, returned iconic sandwich shops

### Key Insights

1. **Text dominates, fusion helps**: Optimal weights (w_text=1.0, w_num=0.7) confirm review content is primary signal, but numeric features (price, location, ratings) provide essential refinement
2. **Cold-start is solvable with content**: 9.2× improvement over random proves viability of content-based bootstrapping before collaborative signals accumulate
3. **Stability matters for production**: K-Means' perfect reproducibility (ARI=1.0) makes it suitable for business reporting and stakeholder communication
4. **Hyperparameter tuning is critical**: Grid search over K∈[3,8] and fusion weights identified optimal configuration (K=4, w_text=1.0, w_num=0.7)
5. **Quality-stability trade-off**: K-Means provides moderate separation (silhouette 0.24) but perfect stability; suitable for interpretable business insights

---

## Limitations & Future Work

### Current Limitations

1. **Temporal validity**: Data extends to 2022; post-pandemic trends (ghost kitchens, delivery dominance) may differ
2. **Geographic generalizability**: Philadelphia-specific patterns (cheesesteaks, BYOB culture) may not transfer to other cities
3. **Review bias**: Yelp reviews skew toward extreme experiences; silent majority of "fine" experiences underrepresented
4. **Recommender recall**: 2.21% Recall@10 is modest; metric is strict (requires exact match in top-10), real-world utility likely higher
5. **Hyperparameter tuning**: Internal validation only (no held-out test set for clustering); limited to K-Means, GMM, Agglomerative

### Immediate Improvements

1. **Transformer embeddings**: Replace NMF with BERT/RoBERTa → richer semantic representation (+10-15% silhouette expected)
2. **Temporal modeling**: Explicitly model time-varying trends via time-series clustering
3. **Expanded tuning**: 5-10× larger hyperparameter grid for UMAP/HDBSCAN/topic count
4. **Image features**: Incorporate restaurant photos via CNN embeddings (better cold-start for photo-heavy, text-light venues)

### Medium-Term Enhancements

5. **Advanced clustering**: Evaluate UMAP+HDBSCAN for fine-grained cluster discovery (expected 2× silhouette improvement, variable-density handling)
6. **Hybrid recommender**: Blend content-based + collaborative filtering once interaction data accumulates (2-3× Recall@10 improvement expected)
7. **Active learning**: Strategically elicit initial ratings (max entropy sampling across clusters) to accelerate cold-start transition
8. **Explainable AI**: Add LIME/SHAP explanations for recommendations ("30% cuisine match, 25% price fit, 20% same neighborhood...")

### Research Directions

9. **Multi-view clustering**: Formally treat text/numeric/graph as separate but related modalities
10. **Cold-start bandits**: Frame as exploration-exploitation problem using contextual bandits
11. **Fairness-aware clustering**: Ensure new/small restaurants aren't systematically under-recommended
12. **Causal inference**: Distinguish correlation (co-reviewed) from causation (restaurant A influenced trying B)

## Broader Applicability

While validated on Philadelphia restaurants, this methodology applies to any recommendation domain with cold-start challenges:

- **E-commerce**: New products without reviews
- **Job matching**: New applicants without application history
- **Content platforms**: New articles/videos without engagement data
- **Healthcare**: New patients without medical history (recommend specialists)
- **Real estate**: New listings without viewing history

**Core principle**: Use intrinsic content features to bootstrap similarity measures until behavioral data accumulates.

This project demonstrates that **cold-start is not insurmountable**. By combining topic modeling, feature fusion, and multiple clustering approaches, a recommender system was built that:
- Requires no labeled training data (fully unsupervised)
- Works immediately for new entities (restaurants and users)
- Achieves non-trivial accuracy (9.2× better than random)
- Balances focused recommendations with diversity (entropy 0.26)
- Provides interpretable explanations (cluster membership, topic alignment)
- Achieves perfect reproducibility (K-Means stability ARI=1.0)

The methodology transfers to any domain with cold-start challenges (e-commerce, job matching, content platforms) where intrinsic content features can bootstrap similarity measures until behavioral data accumulates.

---

## Project Structure & Reproducibility

### Repository Contents

```
unsupervised/
├─ philly_restaurants_streamlined.ipynb  # Main notebook (EDA → modeling → evaluation)
├─ README.md                              # This file
└─ Yelp JSON/
   └─ yelp_dataset/
      ├─ yelp_academic_dataset_business.json
      └─ yelp_academic_dataset_review.json
```

### Dependencies

**Core**: numpy, pandas, scikit-learn, matplotlib, seaborn, nltk

The notebook downloads NLTK's VADER lexicon automatically on first run. All other dependencies are standard scientific Python libraries.

### Reproducibility

- **Random seeds**: All stochastic operations use `RANDOM_STATE=42` for reproducibility
- **Runtime**: ~5-10 minutes on standard laptop (MacBook Pro, 16GB RAM)
- **Platform**: Tested on macOS (Darwin 24.6.0); should work on Linux/Windows with Python 3.8+
- **Notebook execution**: Run cells sequentially from top to bottom (dependencies between cells)
- **Hyperparameter tuning**: Includes grid search over K∈[3,8], fusion weights, GMM covariance types

### To Replicate

1. Clone repository: `git clone https://github.com/shaikh23/Philadelphia-recommender`
2. Download [Yelp Open Dataset](https://www.yelp.com/dataset/download) and extract to `./Yelp JSON/yelp_dataset/`
3. Open `philly_restaurants_streamlined.ipynb` in Jupyter/VS Code
4. Run all cells sequentially
5. Observe clustering results, topic modeling outputs, and cold-start recommender demos

---

## Acknowledgments

- **Data**: Yelp Inc. (Yelp Open Dataset for academic use)
- **NLP**: NLTK VADER sentiment analyzer
- **Topic modeling**: NMF (Non-negative Matrix Factorization) via scikit-learn
- **Clustering**: K-Means, GMM, Agglomerative Clustering (scikit-learn)
- **Scientific stack**: NumPy, pandas, matplotlib, seaborn

---

## License & Data Terms

This project uses the Yelp Open Dataset under Yelp's terms for research/non-commercial use. Code in this repository is provided for educational purposes. See [Yelp Dataset License](https://www.yelp.com/dataset/download) for data usage terms.
