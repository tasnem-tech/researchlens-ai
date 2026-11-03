
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }
    h1 {
        font-weight: 800 !important;
    }
    .stMetric {
        background: #f5f7fb;
        padding: 16px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 ResearchLens AI")

st.caption(
    "Explore datasets, audit data quality, "
    "and benchmark classification models "
    "with leakage-aware pipelines."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Dataset")

    uploaded = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
    )

    st.caption(
        "Runs locally or on your chosen Streamlit deployment. "
        "Do not upload confidential data to untrusted hosts."
    )

# --------------------------------------------------
# DEMO DATASET
# --------------------------------------------------

@st.cache_data
def demo():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)

    df = data.frame.copy()

    df["target"] = df["target"].map({
        0: "class_0",
        1: "class_1",
    })

    return df


try:
    df = pd.read_csv(uploaded) if uploaded else demo()

except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()


if df.empty:
    st.warning("The dataset is empty.")
    st.stop()


st.info(
    "Using a built-in example dataset."
    if uploaded is None
    else f"Loaded {uploaded.name}"
)

# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Data quality",
    "Visual explorer",
    "Model benchmark",
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

with tabs[0]:

    a, b, c, d = st.columns(4)

    a.metric("Rows", f"{len(df):,}")

    b.metric("Columns", len(df.columns))

    c.metric(
        "Missing cells",
        f"{int(df.isna().sum().sum()):,}",
    )

    d.metric(
        "Duplicate rows",
        f"{int(df.duplicated().sum()):,}",
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.download_button(
        label="Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# TAB 2: DATA QUALITY
# --------------------------------------------------

with tabs[1]:

    st.subheader("Data Quality Audit")

    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "missing_pct": (
            100 * df.isna().mean()
        ).round(2).values,
        "unique": df.nunique(dropna=True).values,
    })

    st.dataframe(
        audit,
        use_container_width=True,
    )

    # NEW FEATURE: DOWNLOAD DATA QUALITY REPORT

    st.download_button(
        label="📥 Download Data Quality Report",
        data=audit.to_csv(index=False).encode("utf-8"),
        file_name="researchlens_data_quality_report.csv",
        mime="text/csv",
    )

    st.plotly_chart(
        px.bar(
            audit,
            x="column",
            y="missing_pct",
            title="Missing values by column (%)",
        ),
        use_container_width=True,
    )

    st.warning(
        "Potential leakage, duplicated entities, "
        "class imbalance and temporal dependencies "
        "require domain-specific checks beyond "
        "this automated audit."
    )

# --------------------------------------------------
# TAB 3: VISUAL EXPLORER
# --------------------------------------------------

with tabs[2]:

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric:

        x = st.selectbox(
            "Numeric variable",
            numeric,
        )

        color = st.selectbox(
            "Group by (optional)",
            ["None"] + df.columns.tolist(),
        )

        st.plotly_chart(
            px.histogram(
                df,
                x=x,
                color=None if color == "None" else color,
                nbins=35,
                title=f"Distribution of {x}",
            ),
            use_container_width=True,
        )

    else:
        st.info("No numeric columns available.")

# --------------------------------------------------
# TAB 4: MODEL BENCHMARK
# --------------------------------------------------

with tabs[3]:

    target = st.selectbox(
        "Classification target",
        df.columns.tolist(),
        index=len(df.columns) - 1,
    )

    features = [
        c for c in df.columns
        if c != target
    ]

    work = df[
        features + [target]
    ].dropna(subset=[target]).copy()

    counts = work[target].value_counts()

    st.write("Target distribution")

    st.dataframe(
        counts.rename("count")
    )

    if len(counts) < 2 or len(counts) > 20:

        st.warning(
            "Select a target with 2–20 classes."
        )
        st.stop()

    if len(work) < 30 or counts.min() < 3:

        st.warning(
            "Need at least 30 rows and "
            "3 samples in every class."
        )
        st.stop()

    if not features:

        st.warning("No predictor columns.")
        st.stop()

    st.caption(
        "Train/test split is stratified. "
        "Imputation, scaling and encoding "
        "are fitted only on training data."
    )

    if st.button(
        "Train and evaluate",
        type="primary",
    ):

        X = work[features]

        y = work[target].astype(str)

        numeric = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical = [
            c for c in features
            if c not in numeric
        ]

        prep = ColumnTransformer([
            (
                "num",
                Pipeline([
                    (
                        "impute",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scale",
                        StandardScaler(),
                    ),
                ]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "impute",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encode",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]),
                categorical,
            ),
        ])

        try:

            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )

            results = []

            models = [
                (
                    "Logistic regression",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
                (
                    "Random forest",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]

            for name, model in models:

                pipe = Pipeline([
                    ("preprocess", prep),
                    ("model", model),
                ])

                pipe.fit(Xtr, ytr)

                pred = pipe.predict(Xte)

                results.append({
                    "Model": name,
                    "Balanced accuracy":
                        balanced_accuracy_score(
                            yte, pred
                        ),
                    "Macro F1":
                        f1_score(
                            yte,
                            pred,
                            average="macro",
                            zero_division=0,
                        ),
                })

                with st.expander(
                    f"{name}: detailed report"
                ):

                    st.text(
                        classification_report(
                            yte,
                            pred,
                            zero_division=0,
                        )
                    )

                    labels = sorted(y.unique())

                    st.plotly_chart(
                        px.imshow(
                            confusion_matrix(
                                yte,
                                pred,
                                labels=labels,
                            ),
                            x=labels,
                            y=labels,
                            text_auto=True,
                            labels={
                                "x": "Predicted",
                                "y": "Actual",
                            },
                            title="Holdout confusion matrix",
                        ),
                        use_container_width=True,
                    )

            st.dataframe(
                pd.DataFrame(results).style.format({
                    "Balanced accuracy": "{:.3f}",
                    "Macro F1": "{:.3f}",
                }),
                use_container_width=True,
            )

            st.caption(
                "Exploratory benchmark only. "
                "Random splitting is unsuitable "
                "for some temporal, grouped or "
                "spatial datasets. "
                "Do not use results as clinical "
                "or safety decisions."
            )

        except ValueError as e:

            st.error(
                f"Unable to train: {e}"
            )
