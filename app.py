import streamlit as st
import pickle
import numpy as np

# Load the artifacts (model & data)
model = pickle.load(open('model.pkl', 'rb'))
books_name = pickle.load(open('books_name.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

# App Title
st.title("📚 Book Recommendation App")
st.write("Enter the name of your favorite book to get similar book recommendations!")

# User Input (Text Box)
book_name = st.text_input("Enter Book Name (Case Sensitive):")

# Recommend Button
if st.button("Recommend Similar Books"):
    if book_name in book_pivot.index:
        # Recommendation Logic
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
        )

        st.subheader(f"Recommended Books for '{book_name}':")
        for i in range(1, len(suggestion[0])):  # Skipping the first (same book itself)
            st.write(f"➡️ {book_pivot.index[suggestion[0][i]]}")
    else:
        st.error("❌ Book not found in dataset. Please check the name (case sensitive).")
