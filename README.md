# 🔬 ResearchLens AI

**An interactive data exploration and machine-learning benchmarking dashboard.**

ResearchLens AI helps analysts, students and researchers inspect CSV datasets, detect basic quality issues, visualise distributions and compare two baseline classification models. Built with Python, Streamlit, Plotly and scikit-learn.

> **Status:** Portfolio demo. Not a validated scientific or clinical decision-support system.

## Features
- 📊 Overview with dataset statistics and CSV preview
- 🧹 Missing-value, duplicate and data-type audit
- 📈 Interactive distribution charts
- 🧠 Logistic Regression vs Random Forest comparison
- 🔒 Leakage-aware preprocessing: imputation and encoding fitted on training data only
- 🎯 Balanced accuracy, macro F1, classification reports and confusion matrices
- 🧪 Built-in example dataset so the demo works immediately

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload your own CSV using the sidebar, or explore the built-in example.

## Project structure

```text
researchlens-ai/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Responsible use and limitations
The tool uses a stratified random train/test split and is meant for introductory classification benchmarks. It does **not** implement grouped/temporal validation, nested hyperparameter selection, class-sensitive threshold optimisation, or systematic leakage audits. In small or severely imbalanced datasets, performance estimates can be unstable. Verify feature provenance, privacy rights and evaluation design before drawing conclusions.

## Roadmap
- [ ] Grouped and temporal cross-validation
- [ ] Exportable audit report
- [ ] Regression workflow
- [ ] Explainability with SHAP
- [ ] Automated tests and CI

## Portfolio description
**ResearchLens AI | Interactive data quality and machine-learning benchmarking dashboard built with Python, Streamlit and scikit-learn.**

## License
MIT. See [LICENSE](LICENSE).
