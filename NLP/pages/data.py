import streamlit as st
import pandas as pd
import re

st.title("📂 Upload & Analisis Dataset Berita")

# ===========================
# DOWNLOAD TEMPLATE BUTTON
# ===========================
template_path = "template/Meta Data.xlsx"

try:
    with open(template_path, "rb") as f:
        template_data = f.read()

    st.download_button(
        label="📥 Download Template Dataset",
        data=template_data,
        file_name="Meta Data Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except FileNotFoundError:
    st.error("⚠️ File template tidak ditemukan. Pastikan file berada di folder `template/Meta Data.xlsx`.")

st.write("---")


# Upload File
uploaded = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded:
    try:
        df = pd.read_excel(uploaded, sheet_name="Fix")
        st.success("✅ File berhasil dimuat!")
        
        st.write("📌 Kolom ditemukan:", df.columns.tolist())


        possible_text_columns = ["Headline", "Header", "Judul", "Text", "Isi", "Body"]
        text_col = next((col for col in possible_text_columns if col in df.columns), None)

        if text_col is None:
            st.error("❌ Tidak ditemukan kolom headline seperti: Headline, Header, Judul, Text, Body")
            st.stop()

        st.info(f"📝 Kolom digunakan sebagai teks: **{text_col}**")

        label_col = "Label" if "Label" in df.columns else None


        df["Cleaned_Text"] = df[text_col].astype(str).apply(
            lambda x: re.sub(r'\[?\s*(HOAX|VALID)\s*\]?', '', x, flags=re.IGNORECASE).strip()
        )


        df["_Extracted"] = df[text_col].astype(str).apply(
            lambda x: re.search(r'(HOAX|VALID)', x, flags=re.IGNORECASE).group(1).upper()
            if re.search(r'(HOAX|VALID)', x, flags=re.IGNORECASE)
            else None
        )


        if label_col:
            df["Final_Label"] = df["_Extracted"].fillna(df[label_col])
        else:
            df["Final_Label"] = df["_Extracted"].fillna("UNKNOWN")


        df["Final_Label"] = df["Final_Label"].astype(str).str.upper().str.strip()

        # 🔥 Tambahan penting — deteksi label angka
        df["Final_Label"] = df["Final_Label"].replace({
            "0": "HOAX",
            "1": "VALID",
            "HOAX 1": "HOAX",
            "VALID 1": "VALID"
        })


        label_map = {"HOAX": 0, "VALID": 1}
        df["Label_Num"] = df["Final_Label"].map(label_map).fillna(-1).astype(int)


        st.subheader("📊 Preview Data Cleaned")
        st.dataframe(df[[text_col, "Final_Label", "Label_Num", "Cleaned_Text"]].head(20))

        st.markdown("""
        ### 🏷️ Keterangan Label
        - **0 = HOAX**
        - **1 = VALID**
        - **-1 = UNKNOWN (label tidak terdeteksi)**
        """)

        st.subheader("📌 Tabel Analisis Dataset")
        st.dataframe(df[[text_col, "Cleaned_Text", "Final_Label", "Label_Num"]])


        st.write("🔍 Distribusi Label:")
        st.bar_chart(df["Final_Label"].value_counts())


        processed_file = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Data Cleaning (.csv)",
            data=processed_file,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")

else:
    st.warning("📁 Harap unggah file dataset terlebih dahulu.")
