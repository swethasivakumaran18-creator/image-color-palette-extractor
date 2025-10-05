
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import io

st.title("🎨 Image Color Palette Extractor")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # User chooses number of colors
    k = st.slider("Number of Colors", 2, 10, 5)

    # Convert to numpy
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)

    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    st.subheader("Dominant Colors (RGB Values):")
    st.write(colors)

    # Plot Palette
    fig, ax = plt.subplots(figsize=(8,2))
    for i, color in enumerate(colors):
        ax.fill_between([i, i+1], 0, 1, color=color/255)
    ax.axis("off")

    st.pyplot(fig)
