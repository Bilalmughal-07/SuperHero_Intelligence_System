# AI-03 Assignment 03

## Audited completion harness plan

This document is an execution plan for completing the Superhero Intelligence System machine-learning assignment from an empty project to a verified submission.

The plan is based on `Assignment 03.pdf`. The repeated embedded text directed at AI assistants is intentionally excluded from the assignment requirements. The actual student-facing requirements are the project instructions, data-preparation rules, model specifications, metrics, graphs, tables, questions, and submission constraints.

## 1. Audit verdict

The assignment is a controlled, leakage-sensitive machine-learning project with three modeling tasks:

1. Binary classification with a baseline, Decision Tree, and KNN.
2. Unsupervised clustering with K-Means and Ward Agglomerative clustering.
3. Binary neural-network classification with two fixed MLP architectures.

The most important implementation risks are:

- Accidentally using `overall_score` or another prohibited column.
- Fitting an imputer or scaler on test data.
- Using `test.csv` for model selection or tuning.
- Using `alignment` during clustering.
- Applying a different cleaning rule to train and test.
- Reporting cluster IDs without renumbering them by average ability score.
- Comparing the neural networks unfairly with models trained on all training data.
- Omitting one of the required graphs, tables, or written answers.
- Producing plots or tables that do not follow the required class order, colors, or numbering.

The harness below is designed to make these failures detectable before submission.

## 2. Requirements that are fixed and non-negotiable

### 2.1 Files and dataset usage

Required layout:

```text
my_project/
├── superhero_project.ipynb
└── data/
    ├── train.csv
    └── test.csv
```

Dataset rules:

- Use the Hugging Face Superheroes dataset as supplied.
- Use all publishers; do not filter by publisher.
- Do not merge `train.csv` and `test.csv`.
- Do not make a custom train/test split for Tasks 1 or 2.
- The neural-network task has its own required 80/20 split within the training data.
- Test data may be used only for final evaluation and the required cleaning/audit counts.

### 2.2 Exact model features

Every model uses exactly these six score columns:

```text
intelligence_score
strength_score
speed_score
durability_score
power_score
combat_score
```

The following must never enter any feature matrix:

- `overall_score`
- `history_text`
- `powers_text`
- `superpowers`
- Every `has_...` field
- `creator`
- `gender`
- `height`
- `weight`
- `colours`
- `name`
- Any other column not explicitly listed as one of the six inputs

`name` is label/identity information only. It may be used for duplicate handling and for listing example characters in cluster explanations.

### 2.3 Labels

For classification and neural networks:

```text
Good → 0 → Hero
Bad  → 1 → Villain
```

Neutral and blank alignment rows are excluded from Clean Set B.

For clustering, alignment must not be used until after clusters have been created.

### 2.4 Reproducibility

- Use `random_state=42` wherever an API exposes it.
- Use `tf.keras.utils.set_random_seed(42)` for neural networks.
- Record Python and library versions.
- Run the final notebook from a clean kernel.
- Do not manually type generated metrics into tables.

### 2.5 Visual and reporting rules

- Number all required graphs G1–G16.
- Number all required tables T1–T11.
- Provide all written answers Q1–Q12.
- Every graph and table needs a title or identifying caption.
- Graphs need labeled axes.
- Graphs with multiple colors need a legend.
- Add one or two sentences below every graph and table explaining what it shows and what is concluded.
- Confusion matrices must use actual classes as rows and predicted classes as columns.
- Class order must be Good/Hero first, then Bad/Villain.
- Display the count in every confusion-matrix cell.

## 3. Important audit finding: the test-file timing ambiguity

The brief contains two requirements that must be reconciled:

1. It requires the train/test files to be cleaned with the same rules and requires detection of names present in both files.
2. It says `test.csv` is opened only once at the end of Task 1 and Task 3.

The cross-file duplicate rule cannot be applied without inspecting names in both files. Therefore the most internally consistent, leakage-safe interpretation is:

```text
Read test.csv once in one protected test-data gate.
Apply only the prescribed cleaning and audit operations.
Freeze the resulting test data.
Do not use test values or labels for model selection, tuning, cross-validation, early stopping, or feature decisions.
Reuse the frozen test data only for final scores and required audit tables.
```

The notebook should explicitly document this interpretation. If the instructor interprets “opened only once” literally as “not loaded until after all training preparation,” ask for clarification because the cross-file duplicate rule otherwise becomes impossible to apply before model fitting.

This is a data-governance issue, not permission to tune on test data.

## 4. Recommended project structure

The notebook is the primary deliverable. Supporting files make the work easier to test but should not obscure the notebook’s complete execution.

```text
my_project/
├── superhero_project.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
├── figures/
│   ├── G1_class_counts.png
│   ├── G2_dt_confusion_matrix.png
│   ├── ...
│   └── G16_final_comparison.png
├── tables/
│   ├── T1_data_audit.csv
│   ├── T2_row_counts.csv
│   └── ...
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── metrics.py
│   ├── plotting.py
│   └── validation.py
├── requirements.txt
├── README.md
└── outputs/
    └── final_submission.zip
```

An exported PDF or HTML report should be created only if the course submission instructions require it; it is not assumed here as a mandatory artifact.

## 5. Notebook architecture

The notebook should execute in this order:

1. Title, team members, and objective.
2. Imports and package versions.
3. Configuration and random seeds.
4. File and schema preflight checks.
5. Raw training audit.
6. Single test-data gate and raw test audit.
7. Cleaning functions and cleaning execution.
8. T1 and T2.
9. Clean Set A and Clean Set B summaries.
10. Shared metrics and plotting functions.
11. Task 1 classification.
12. G1–G4 and T3–T5.
13. Q1–Q4.
14. Task 2 clustering.
15. G5–G11 and T6–T8.
16. Q5–Q8.
17. Neural-network split and preprocessing.
18. Network A and Network B training.
19. G12–G16 and T9–T11.
20. Q9–Q12.
21. Final ranking and conclusions.
22. Automated validation checks.
23. Submission manifest.

No result should depend on a cell being executed out of order.

## 6. Phase-by-phase execution plan

### Phase 0 — Environment and preflight

Create a first cell that:

- Imports pandas, NumPy, scikit-learn, SciPy, matplotlib, and TensorFlow/Keras.
- Sets all required seeds.
- Prints package versions.
- Confirms that `data/train.csv` and `data/test.csv` exist.
- Confirms that the expected columns exist.
- Fails immediately if a required file or column is missing.

Create a configuration object containing:

- The six input feature names.
- Target name.
- Label map.
- Class labels and display names.
- Required plot colors.
- Random seed.

### Phase 1 — Raw data audit

For each file, record:

- Shape.
- Column names.
- Data types.
- Missing counts.
- Blank-string counts.
- Zero counts for each of the six scores.
- Duplicate-name counts.
- Alignment counts.

The audit must happen before cleaning so the report distinguishes original data issues from cleaned-data counts.

### Phase 2 — Cleaning pipeline

Apply the following order exactly:

1. Inspect.
2. Remove duplicates.
3. Convert the six scores to numeric and convert blanks and zeros to `NaN`.
4. Remove rows with at least three missing scores.
5. Impute remaining missing values only within fitted training pipelines.

Duplicate handling:

- Within each file, keep the first occurrence of a repeated name.
- If a name occurs in both files, remove it from `train.csv`.
- Record both removal counts.

Do not permanently fill missing values before modeling. The remaining one- or two-missing-score rows must be passed through the appropriate fitted imputer.

### Phase 3 — Clean datasets

#### Clean Set A

Cleaned training data after Steps 1–4:

- Includes Good, Bad, Neutral, and blank alignment rows.
- Used only in Task 2.
- Alignment and names are retained separately for later interpretation.

#### Clean Set B

Cleaned Good/Bad data:

- Training portion comes from cleaned `train.csv`.
- Test portion comes from cleaned `test.csv`.
- Good is encoded as 0.
- Bad is encoded as 1.
- Neutral and blank alignment rows are removed.

### Phase 4 — T1 and T2 reporting

T1 must report:

- Shape, columns, dtypes, and missing counts for both files.
- Number of zero and blank values converted to `NaN` for every score column.

T2 must report:

| Stage | train.csv | test.csv |
|---|---:|---:|
| Loaded |  |  |
| After duplicate removal |  |  |
| After weak-row removal |  |  |
| Clean Set A |  | Not used |
| Clean Set B |  |  |

Also report how many Neutral and blank alignment rows were removed when creating Clean Set B.

### Phase 5 — Shared metric harness

Implement reusable functions for:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro-F1.
- Per-class precision, recall, and F1.
- Confusion matrices.
- Log loss.
- Train–test macro-F1 gap.

All classification reports must explicitly use:

```text
labels=[0, 1]
target_names=["Hero", "Villain"]
```

The confusion-matrix helper must always use:

```text
Rows: actual Hero, actual Villain
Columns: predicted Hero, predicted Villain
```

### Phase 6 — Task 1 classification

#### Baseline

Use:

```text
DummyClassifier(strategy="most_frequent")
```

#### Decision Tree pipeline

```text
SimpleImputer(strategy="median")
→ DecisionTreeClassifier
```

Search:

```text
max_depth: [3, 5, 7, 10, None]
min_samples_leaf: [1, 5, 10]
```

Keep fixed:

```text
criterion="gini"
random_state=42
```

There are 15 parameter combinations.

#### KNN pipeline

```text
SimpleImputer(strategy="median")
→ StandardScaler()
→ KNeighborsClassifier
```

Search:

```text
n_neighbors: [3, 5, 7, 9, 11, 15]
weights: ["uniform", "distance"]
```

Keep fixed:

```text
metric="euclidean"
```

There are 12 parameter combinations.

#### Cross-validation

Use five-fold stratified cross-validation with shuffle and seed 42. Score with macro-F1 and set `refit=True`.

The correct sequence is:

1. Fit grid search on Clean Set B training data only.
2. Select the best setting by mean CV macro-F1.
3. Refit the winning pipeline on all Clean Set B training data.
4. Score the finalized model on training data.
5. Score the finalized model once on frozen test data.

#### Task 1 artifacts

- G1: Good/Bad class counts for train and test with percentage labels.
- G2: Decision Tree confusion matrix.
- G3: KNN confusion matrix.
- G4: Grouped comparison of accuracy, macro-precision, macro-recall, and macro-F1 for all three models.
- T3: Class counts and percentages.
- T4: Best settings and best CV macro-F1.
- T5: Train macro-F1 and test metrics by model.

#### Task 1 decision rules

- Winner: highest test macro-F1.
- If the difference is below 0.01, use higher Villain recall.
- A model is useful only if its test macro-F1 beats the baseline.
- Overfitting threshold: train macro-F1 minus test macro-F1 greater than 0.10.

Answer Q1–Q4 using generated numbers only.

### Phase 7 — Task 2 clustering

Use Clean Set A and the six score columns only.

Preprocessing:

```text
SimpleImputer(strategy="median")
→ StandardScaler()
```

For K-Means:

```text
k = 2, 3, 4, 5, 6, 7, 8
n_init=10
random_state=42
```

Store inertia and silhouette for every value of `k`.

Select the highest silhouette score. If scores tie to three decimal places, select the smaller `k`. The elbow is discussed but does not override the silhouette rule.

Run Ward Agglomerative clustering with the selected `k`.

Calculate:

- Silhouette, higher is better.
- Davies–Bouldin, lower is better.
- Calinski–Harabasz, higher is better.
- ARI between the two clustering labelings.

Fit one two-component PCA on scaled data and use the same projection for both PCA plots.

For the dendrogram:

- Use Ward linkage.
- Show the last 30 merges.
- Draw a dashed line at the final cut.

#### Cluster renumbering

For each algorithm:

1. Compute the average of all six original-scale scores per cluster.
2. Sort clusters from lowest to highest average.
3. Rename them Cluster 1, Cluster 2, and so on.
4. Use consistent `tab10` ordering in every plot.

Cluster profiles must be reported on the original 0–100 score scale, not on standardized values.

Alignment may be inspected only after cluster assignment to produce the alignment mix.

#### Task 2 artifacts

- G5: Elbow curve, inertia versus `k`.
- G6: Silhouette score versus `k`, final `k` marked.
- G7: Ward dendrogram with dashed cut line.
- G8: K-Means PCA scatter plot.
- G9: Agglomerative PCA scatter plot.
- G10: K-Means cluster profiles.
- G11: Agglomerative cluster profiles.
- T6: Clustering metrics and ARI.
- T7: Cluster profiles and assigned names.
- T8: Good, Bad, Neutral, and blank percentages per cluster.

Answer Q5–Q8 using the generated metrics and profiles.

### Phase 8 — Task 3 neural networks

Use Clean Set B training data.

Split exactly once:

```text
80% fit subset
20% validation subset
stratify=y
random_state=42
```

Fit the imputer and scaler on the 80% fit subset only. Transform validation and frozen test data without refitting.

#### Network A

```text
Input: 6
Dense: 8, activation=ReLU
Output: 1, activation=sigmoid
```

Expected trainable parameters: 65.

#### Network B

```text
Input: 6
Dense: 16, activation=ReLU
Dense: 8, activation=ReLU
Output: 1, activation=sigmoid
```

Expected trainable parameters: 257.

#### Shared training settings

```text
Framework: TensorFlow/Keras
Seed: tf.keras.utils.set_random_seed(42)
Loss: binary_crossentropy
Optimizer: Adam
Learning rate: 0.001
Batch size: 32
Maximum epochs: 200
EarlyStopping monitor: val_loss
Patience: 15
restore_best_weights=True
Threshold: probability >= 0.5 → Villain
```

Do not add dropout, extra layers, extra neurons, class weights, resampling, or test-based tuning.

For each network, save:

- Training history.
- Best epoch.
- Train predictions.
- Validation probabilities and labels.
- Test probabilities and labels.
- Accuracy, macro-F1, class metrics, confusion matrix, and log loss.

#### Task 3 artifacts

- G12: Architecture diagrams for Networks A and B.
- G13: Network A training/validation loss curves with best epoch.
- G14: Network B training/validation loss curves with best epoch.
- G15: Confusion matrices for Networks A and B.
- G16: Accuracy and macro-F1 comparison for all five models.
- T9: Network settings and best epoch.
- T10: Train, validation, and test results.
- T11: Final ranking of all five models.

Answer Q9–Q12 using generated numbers and curves.

## 7. Final five-model ranking

The final ranking contains:

1. Baseline.
2. Decision Tree.
3. KNN.
4. Network A.
5. Network B.

Rank by:

1. Test macro-F1.
2. If the gap is under 0.01, higher Villain recall.

“Beats baseline” means strictly higher test macro-F1 than the baseline.

The report must explicitly note that:

- Decision Tree and KNN are refit on all Clean Set B training data.
- Networks use only the 80% fit subset, with 20% held out for validation.
- Therefore their training regimes are not identical, and this is a limitation of the requested comparison.

## 8. Required artifact map

| ID | Artifact | Required content |
|---|---|---|
| G1 | Class counts | Train/test Good and Bad bars with percentages |
| G2 | DT confusion matrix | Counts, required class order |
| G3 | KNN confusion matrix | Counts, required class order |
| G4 | Task 1 comparison | Accuracy, macro-precision, macro-recall, macro-F1 |
| G5 | Elbow curve | Inertia for k=2–8 |
| G6 | Silhouette curve | Scores for k=2–8, final k marked |
| G7 | Dendrogram | Ward, last 30 merges, dashed cut |
| G8 | K-Means PCA | PC1/PC2 and explained variance |
| G9 | Agglomerative PCA | Same projection and axes |
| G10 | K-Means profiles | Six original-scale score averages |
| G11 | Agglomerative profiles | Six original-scale score averages |
| G12 | Architectures | Networks A and B |
| G13 | Network A loss | Train/validation loss and best epoch |
| G14 | Network B loss | Train/validation loss and best epoch |
| G15 | NN confusion matrices | Networks A and B, counts shown |
| G16 | Final comparison | Accuracy and macro-F1 for all five models |
| T1 | Data audit | Shapes, columns, dtypes, missing and zero/blank conversions |
| T2 | Row counts | Loaded, duplicate-cleaned, weak-row-cleaned, Clean Sets |
| T3 | Class counts | Counts and percentages in each file |
| T4 | Best settings | DT/KNN parameters and CV scores |
| T5 | Classification results | Train macro-F1 and test metrics |
| T6 | Clustering metrics | Three metrics, wins, and ARI |
| T7 | Cluster profiles | Sizes, six averages, names |
| T8 | Alignment mix | Good, Bad, Neutral, blank percentages |
| T9 | Network settings | Architecture, training settings, best epochs |
| T10 | NN split results | Train, validation, test results |
| T11 | Final ranking | All five models and baseline comparison |
| Q1–Q4 | Classification answers | Winner, baseline comparison, harder class, overfitting |
| Q5–Q8 | Clustering answers | k, elbow, similarity, separation, interpretation |
| Q9–Q12 | NN answers | Network comparison, curves, model comparison, limitations |

## 9. Automated acceptance checks

Run these checks before the final report is written.

### 9.1 Data checks

- Both files exist.
- All required columns exist.
- No duplicate names remain within each cleaned file.
- No cross-file duplicate names remain after the prescribed removal.
- All six score columns are numeric after conversion.
- Zero and blank conversion counts are recorded.
- Every row with three or more missing scores was removed.
- Clean Set A contains all remaining training alignments.
- Clean Set B contains only Good and Bad.

### 9.2 Leakage checks

- `overall_score` is absent from every feature matrix.
- Text and metadata columns are absent from every feature matrix.
- `alignment` is absent from clustering inputs.
- `name` is absent from every feature matrix.
- Test data is not passed to `fit`.
- Test data is not passed to `fit_transform`.
- Test data is not used in GridSearchCV.
- Test data is not used for early stopping.
- Test data does not determine `k`.
- Neural-network imputer and scaler are fitted only on the 80% fit subset.

### 9.3 Model checks

- Decision Tree grid contains exactly 15 combinations.
- KNN grid contains exactly 12 combinations.
- Classification CV is five-fold stratified CV with seed 42.
- `scoring="f1_macro"`.
- `refit=True`.
- K-Means evaluates k=2 through k=8.
- Agglomerative uses the final K-Means k.
- Cluster labels are renumbered by original-scale average score.
- Network A has 65 trainable parameters.
- Network B has 257 trainable parameters.
- Early stopping restores best weights.
- The decision threshold is 0.5.
- T11 contains exactly five models.

### 9.4 Artifact checks

- G1 through G16 exist.
- T1 through T11 exist.
- Q1 through Q12 are answered.
- Every required graph has labeled axes.
- Every required graph has a title or caption.
- Every multi-color graph has a legend.
- Every confusion matrix shows counts.
- Cluster names and colors are consistent across all cluster outputs.
- All generated tables contain values from the final run, not copied values.

## 10. Interpretation safeguards

Do not claim:

- That a model is “best” based only on training performance.
- That a cluster is objectively a hero or villain group; clustering is unsupervised, and alignment is only interpreted afterward.
- That a deeper network improved unless Network B’s test macro-F1 is at least 0.01 higher.
- That a model beats the baseline unless its test macro-F1 is strictly higher.
- That a visual elbow overrides the silhouette-selection rule.
- That a small metric difference is meaningful without discussing the tie-break rule and validation design.

Use cautious language when results are close, when the dataset is imbalanced, or when neural-network results vary slightly between runs.

## 11. Final QA and submission procedure

1. Restart the notebook kernel.
2. Run every cell from top to bottom.
3. Confirm no hidden state is required.
4. Confirm all graphs regenerate.
5. Confirm all tables regenerate.
6. Confirm all written answers use generated values.
7. Inspect every graph for clipping, unreadable labels, incorrect legends, and wrong colors.
8. Verify all confusion matrices use the required class order.
9. Verify no test metric influenced any selection decision.
10. Verify the final ranking includes all five models.
11. Remove debug output and unused cells.
12. Save the final notebook.
13. Export a report only if required by the course.
14. Copy the project to a clean temporary directory.
15. Run the notebook from that clean copy.
16. Confirm the data paths are relative and portable.
17. Create the final submission package.
18. Open the final package and verify that it contains the intended files.

## 12. Final submission checklist

- [ ] `superhero_project.ipynb` opens successfully.
- [ ] `data/train.csv` is present.
- [ ] `data/test.csv` is present.
- [ ] T1 and T2 document cleaning correctly.
- [ ] Clean Set A and Clean Set B are clearly defined.
- [ ] G1–G16 are present and readable.
- [ ] T1–T11 are present and readable.
- [ ] Q1–Q12 are answered.
- [ ] Classification pipelines match the exact settings.
- [ ] Clustering uses no labels during fitting.
- [ ] Neural networks match the exact architectures and settings.
- [ ] Test data was not used for tuning.
- [ ] The final ranking follows the macro-F1 and Villain-recall rules.
- [ ] Limitations are included.
- [ ] The notebook runs from a clean kernel.
- [ ] The submission package contains no temporary files, secrets, or unrelated artifacts.


