import os
import ollama
import streamlit as st
from PIL import Image

# Constants
UPLOAD_FOLDER = "uploads"
SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text and return the responce in markdown format.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments.
4."""

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize session state
if 'ocr_history' not in st.session_state:
    st.session_state.ocr_history = []
if 'current_ocr' not in st.session_state:
    st.session_state.current_ocr = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def perform_ocr(image_path):
    """Perform OCR on the given image using Llama 3.2-Vision through the Ollama library."""
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': SYSTEM_PROMPT,
                'images': [image_path]
            }]
        )
        return response.get('message', {}).get('content', "")
    except Exception as e:
        st.error(f"An error occurred during OCR processing: {e}")
        return None

# Streamlit app
st.set_page_config(
    page_title="LlamaOCR - Local OCR Tool",
    page_icon="📄",
    layout="wide"
)

st.title("📄 LlamaOCR - Local OCR Tool")
st.write("Effortlessly extract text from images using the power of **Llama 3.2-Vision**. Upload an image to get started!")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.subheader("Uploaded Image:")
    image = Image.open(file_path)
    st.image(image, caption="Your Uploaded Image", use_column_width=True)

    # Perform OCR
    with st.spinner("Performing OCR..."):
        ocr_result = perform_ocr(file_path)

    if ocr_result:
        # Save to history
        st.session_state.ocr_history.append((uploaded_file.name, ocr_result))
        st.session_state.current_ocr = ocr_result

        st.subheader("OCR Result:")
        st.text_area("Extracted Text", ocr_result, height=200)

        # Query interface
        st.subheader("Ask Questions About the Text")
        user_query = st.text_input("Your question:")
        if user_query:
            # Add user question to chat history
            st.session_state.chat_history.append(("user", user_query))

            # Generate response
            context = f"Based on this extracted text:\n{ocr_result}\n\nQuestion: {user_query}"
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{'role': 'user', 'content': context}]
            ).get('message', {}).get('content', "")

            if response:
                st.session_state.chat_history.append(("assistant", response))

            # Display chat history
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.write("You:", message)
                else:
                    st.write("Assistant:", message)
    else:
        st.error("OCR failed. Please try again.")
else:
    st.info("Please upload an image to begin.")

# Display OCR history
if st.session_state.ocr_history:
    st.sidebar.subheader("OCR History")
    history_options = [name for name, _ in st.session_state.ocr_history]
    selected_history = st.sidebar.selectbox("Select a previous OCR result:", history_options)
    for name, result in st.session_state.ocr_history:
        if name == selected_history:
            st.sidebar.text_area("Previous OCR Result", result, height=150)

# Add footer
st.markdown(
    """
    ---
    **LlamaOCR** | Built with ❤️ using [Streamlit](https://streamlit.io) and **Llama 3.2-Vision**.
    """
)