import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re


@st.cache_resource
def load_model():
    save_dir = "saved_finetuning_indobert_model"
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model = BertForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


LABEL_MAP = {
    0: "HOAX",
    1: "VALID"
}


def preprocess(text):
    # Hapus prefix misalnya "[HOAX]" atau "[VALID]" di awal teks
    text = re.sub(r'\[[^\]]+\]\s*', '', text).strip()
    return text


def predict(text):
    cleaned = preprocess(text)

    inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    prob = torch.softmax(logits, dim=1)
    pred = torch.argmax(prob).item()

    return LABEL_MAP[pred], prob.squeeze().tolist()


st.title("🔍 IndoBERT Fake News Detection")
st.write("Masukkan teks berita untuk memeriksa apakah ini **HOAX** atau **VALID**.")

text = st.text_area("Masukkan berita:", height=160)

if st.button("🚀 Prediksi"):
    if text.strip() == "":
        st.warning("⚠ Teks tidak boleh kosong.")
    else:
        label, probs = predict(text)

        prob_percent = round(max(probs) * 100, 2)

        if label == "HOAX":
            st.error(f"❌ Prediksi: **HOAX** ({prob_percent}%)")
        else:
            st.success(f"✅ Prediksi: **VALID** ({prob_percent}%)")

        st.write("\n📊 Probabilitas Model:")
        st.write(f"- HOAX  : {round(probs[0] * 100, 2)}%")
        st.write(f"- VALID : {round(probs[1] * 100, 2)}%")

        # Low confidence warning
        if prob_percent < 60:
            st.warning("⚠ Model kurang yakin. Data ini perlu dicek manual.")


st.write("---")
st.caption("Model: Fine-tuned IndoBERT | Dev by Alfian, Rafael , Firzi ")
