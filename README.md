# Heart Disease Prediction Thesis Workspace

This workspace is now cleaned around one client-ready experiment notebook.

## Main Files

- `notebooks/00_clean_heart_disease_experiments.ipynb`
  - Main notebook to run in VS Code or Jupyter.
  - Organized by clean architecture sections: configuration, data layer, preprocessing, modeling, evaluation, and visualization.
  - Includes tuned hybrids, feature selection analysis, SHAP explainability, and a Feature Tokenizer Transformer baseline.
  - Metrics are generated from code execution, not manually typed into markdown.

- `outputs/clean_notebook/00_clean_heart_disease_experiments.executed.ipynb`
  - Executed notebook with all outputs already rendered.

- `outputs/clean_notebook/00_clean_heart_disease_experiments.html`
  - Printable/shareable HTML version for review.

- `reports/heart_disease_client_technical_report.pdf`
  - Client technical report summarizing the full work, additions, datasets,
    models, hybrid approach, and generated results.

- `reports/heart_disease_arabic_research_report.pdf`
  - Arabic research report covering the full work from the beginning through the
    latest additions and generated results.

- `outputs/clean_notebook/published_work_comparison.csv`
  - Literature comparison table separating published metrics from notebook-generated results.

- `outputs/clean_notebook/research_contribution_matrix.csv`
  - Contribution-positioning table showing how the work addresses research novelty.

- `scripts/generate_clean_architecture_notebook.py`
  - Generator used to rebuild the main notebook if we need to revise structure or wording.

- `scripts/generate_client_technical_report.py`
  - Generator used to rebuild the client report from executed notebook CSV outputs.

- `scripts/generate_arabic_research_report.py`
  - Generator used to rebuild the Arabic research report from executed notebook CSV outputs.

## Data Inputs

Keep these folders available:

- `archive/cardio_train.csv`
  - Kaggle Cardiovascular Disease dataset.

- `data/uci_heart/`
  - UCI processed heart disease files.

- `data/openml_heart_1190/`
  - OpenML Heart Disease Comprehensive dataset.

## Local Run

Use the existing virtual environment if it is already active:

```powershell
.\.venv\Scripts\activate
```

Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Run the notebook from VS Code using the `.venv` kernel.

To regenerate and execute the clean notebook from the terminal:

```powershell
.\.venv\Scripts\python.exe scripts\generate_clean_architecture_notebook.py
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks\00_clean_heart_disease_experiments.ipynb --output 00_clean_heart_disease_experiments.executed.ipynb --output-dir outputs\clean_notebook --ExecutePreprocessor.kernel_name=heart-disease-local --ExecutePreprocessor.timeout=3600
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html outputs\clean_notebook\00_clean_heart_disease_experiments.executed.ipynb --output 00_clean_heart_disease_experiments.html --output-dir outputs\clean_notebook
.\.venv\Scripts\python.exe scripts\generate_client_technical_report.py
```

## Notes

- Old exploratory notebooks, reports, and benchmark artifacts were removed to avoid confusion.
- The `src/heart_disease_prediction/` package is kept as reusable production-style code.
- The client-facing artifact is the clean notebook/HTML, not the deleted drafts.
