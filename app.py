from pathlib import Path
import json, sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from preprocessing import clean_text

st.set_page_config(page_title="Research Abstract Classifier",page_icon="📚",layout="wide")
st.title("📚 Research Abstract Classification")
st.caption("Classify academic abstracts into Science, Business, or Social Science.")
model_path=ROOT/"models"/"best_model.joblib"
info_path=ROOT/"models"/"model_info.json"
results_path=ROOT/"reports"/"model_comparison.csv"
if not model_path.exists() or not results_path.exists():
    st.error("Model artifacts are missing. Run `python src/train.py` from the project directory first.")
    st.stop()
model=joblib.load(model_path)
info=json.loads(info_path.read_text())
results=pd.read_csv(results_path)

st.subheader("Model comparison")
st.write(f"Selected model: **{info['best_model']}**")
st.dataframe(results.style.format({c:"{:.3f}" for c in results.columns if c!="model"}),use_container_width=True)
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(results["model"],results["f1_weighted"])
ax.set_ylabel("Weighted F1"); ax.set_ylim(0,1); ax.tick_params(axis="x",rotation=25)
st.pyplot(fig); plt.close(fig)

st.subheader("Confusion matrix")
cm_files=sorted((ROOT/"reports").glob("confusion_matrix_*.png"))
selected=st.selectbox("Choose model confusion matrix",[f.stem.replace("confusion_matrix_","").replace("_"," ").title() for f in cm_files])
if cm_files:
    chosen=next(f for f in cm_files if f.stem.endswith(selected.lower().replace(" ","_")))
    st.image(str(chosen),use_container_width=True)

st.subheader("Sample counts across classes")
data_path=ROOT/"data"/"abstracts_labeled.csv"
data=pd.read_csv(data_path)
counts=data["label"].value_counts().rename_axis("Class").reset_index(name="Samples")
c1,c2=st.columns([1,2])
with c1: st.dataframe(counts,use_container_width=True,hide_index=True)
with c2:
    fig,ax=plt.subplots(figsize=(6,3)); ax.bar(counts["Class"],counts["Samples"]); ax.set_ylabel("Samples")
    st.pyplot(fig); plt.close(fig)

st.divider()
st.header("Make predictions")
single_tab,batch_tab=st.tabs(["Single abstract upload","Batch CSV prediction"])
with single_tab:
    st.write("Upload a `.txt` file containing one research abstract, or paste the abstract below.")
    uploaded=st.file_uploader("Upload abstract (.txt)",type=["txt"],key="single")
    text=uploaded.getvalue().decode("utf-8",errors="ignore") if uploaded else ""
    text=st.text_area("Abstract text",value=text,height=220)
    if st.button("Predict abstract",type="primary",disabled=not text.strip()):
        pred=model.predict([text])[0]
        st.success(f"Predicted category: **{pred}**")
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba([text])[0]
            st.dataframe(pd.DataFrame({"Class":model.classes_,"Probability":probs}).sort_values("Probability",ascending=False),hide_index=True)
with batch_tab:
    st.write("Upload a CSV containing an `Abstract` column. The app adds `predicted_label` and, when available, class probabilities.")
    batch=st.file_uploader("Upload batch CSV",type=["csv"],key="batch")
    if batch:
        try:
            batch_df=pd.read_csv(batch)
            if "Abstract" not in batch_df.columns:
                st.error("CSV must contain a column named `Abstract`.")
            else:
                valid=batch_df["Abstract"].fillna("").astype(str).str.strip().ne("")
                if not valid.any():
                    st.warning("No non-empty abstracts found.")
                else:
                    preds=model.predict(batch_df.loc[valid,"Abstract"].astype(str))
                    batch_df["predicted_label"]=pd.NA
                    batch_df.loc[valid,"predicted_label"]=preds
                    st.dataframe(batch_df.head(100),use_container_width=True)
                    st.download_button("Download predictions CSV",batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="abstract_predictions.csv",mime="text/csv")
        except Exception as e:
            st.error(f"Could not process CSV: {e}")
