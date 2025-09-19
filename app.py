import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from youtube_analyzer import analyze_youtube_comments
import gzip

@st.cache_resource
def load_model():
    model_path = 'Toxic_Analyzer.pkl.gz'
    with gzip.open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def CheckToxic(text, model):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    return pred, max(proba)

st.set_page_config(page_title="🛡️ Toxic Comment Analyzer", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

def main():
    # --- Main Title Section ---
    st.markdown(
        "<h1 style='text-align: center; color: #4F8EF7;'>🛡️ YouTube Toxic Comment Analyzer</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Analyze YouTube comments for toxicity and visualize them beautifully 🚀</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    model = load_model()

    # --- Sidebar ---
    with st.sidebar:
        st.title("🔹 Navigation")
        selection = st.radio("", ["🏡 Home", "🧪 Analyze Comments", "📊 Visualizations"])

    if selection == "🏡 Home":
        st.markdown(
            """
            <h2 style="text-align: center; color: white;">🏡 Welcome to Toxic Comment Analyzer!</h2>
            <p style="text-align: center;">Analyze YouTube comments for toxicity </p>
            """,
            unsafe_allow_html=True
        )


    elif selection == "🧪 Analyze Comments":
        st.subheader("🎥 Analyze YouTube Comments")
        youtube_url = st.text_input("🔗 Enter YouTube Video Link:")

        if st.button("🚀 Start Analyzing"):
            if not youtube_url.strip():
                st.warning("⚠️ Please paste a valid YouTube video URL.")
            else:
                with st.spinner('⏳ Fetching and analyzing comments...'):
                    try:
                        results = analyze_youtube_comments(youtube_url, model)

                        if not results.empty:
                            st.success(f"✅ Successfully analyzed {len(results)} comments!")
                            
                            # ----- Stats Section -----
                            total_comments = len(results)
                            toxic_comments = (results['Prediction'] == 'Toxic').sum()
                            non_toxic_comments = (results['Prediction'] == 'Non-Toxic').sum()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("💬 Total Comments", total_comments)
                            col2.metric("☣️ Toxic Comments", toxic_comments)
                            col3.metric("🌿 Non-Toxic Comments", non_toxic_comments)

                            st.markdown("---")
                            st.subheader("📋 Analyzed Comments")
                            st.dataframe(results, use_container_width=True)

                            # Save to session
                            st.session_state['results'] = results

                            # Download CSV Button
                            csv = results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Results as CSV",
                                data=csv,
                                file_name='analyzed_comments.csv',
                                mime='text/csv',
                                key='download-csv'
                            )
                        else:
                            st.warning("😕 No comments found or couldn't fetch.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    elif selection == "📊 Visualizations":
        st.subheader("📊 Visualizations")

        if 'results' not in st.session_state:
            st.warning("⚠️ Please analyze some comments first!")
        else:
            results = st.session_state['results']

            viz_option = st.selectbox("📈 Choose a Visualization:", 
                                      ["Toxic Comment Ratio (Pie Chart)", 
                                       "Toxicity Confidence (Distribution)", 
                                       "Toxic vs Non-Toxic Count"])

            if viz_option == "Toxic Comment Ratio (Pie Chart)":
                if 'Prediction' not in results.columns:
                    st.error("❌ 'Prediction' column missing. Analyze first!")
                else:
                    toxic_counts = results['Prediction'].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    colors = ['#FF6B6B', '#4CAF50']
                    ax.pie(toxic_counts, labels=toxic_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={"edgecolor":"black"})
                    ax.axis('equal')
                    plt.title("Toxic vs Non-Toxic Comments", fontsize=16)
                    st.pyplot(fig)

            elif viz_option == "Toxicity Confidence (Distribution)":
                if 'Confidence' not in results.columns:
                    st.error("❌ 'Confidence' column missing. Analyze first!")
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(results['Confidence'], bins=20, kde=True, color='#7E57C2')
                    plt.title("Distribution of Toxicity Confidence", fontsize=16)
                    plt.xlabel("Confidence Level")
                    plt.ylabel("Number of Comments")
                    st.pyplot(fig)

            elif viz_option == "Toxic vs Non-Toxic Count":
                if 'Prediction' not in results.columns:
                    st.error("❌ 'Prediction' column missing. Analyze first!")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x='Prediction', data=results, palette='coolwarm')
                    plt.title("Toxic vs Non-Toxic Comment Count", fontsize=16)
                    plt.xlabel("Type")
                    plt.ylabel("Count")
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
