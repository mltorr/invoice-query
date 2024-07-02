import streamlit as st
from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from io import BytesIO

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_prompt():
    input_prompt = """
                   You are a financial advisor with an expertise in understanding invoices.
                   You will receive input images as invoices &
                   you will have to answer questions based on the input image.
                   If you don't know the answer, please refrain from speculating.
                   """
    return input_prompt

def generate_response_llm(input_question, prompt, image):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([input_question, prompt, image])
    return response.text

def convert_pdf_to_images(pdf_data):
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
        temp.write(pdf_data)
        temp_path = temp.name

    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(temp_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=170)
            img_temp_path = Path(temp_dir) / f"page_{page_num}.jpg"
            pix.save(str(img_temp_path), jpg_quality=98)
            img_data = img_temp_path.read_bytes()
            images.append(img_data)

    return images

def main():
    st.set_page_config(page_title="Invoice Query v0.1")

    st.title("Invoice Query v0.1")

    # Initialize session state for the input prompt
    if "input_prompt" not in st.session_state:
        st.session_state.input_prompt = ""

    user_question = st.text_input("Input prompt", value=st.session_state.input_prompt, key="input")

    st.sidebar.title("Invoice File")

    # Button to set the input prompt
    if st.sidebar.button("Sample Prompt 1"):
        st.session_state.input_prompt = "Please parse all the information on the invoice."
        st.experimental_rerun()

    if st.sidebar.button("Sample Prompt 2"):
        st.session_state.input_prompt = "Please parse the invoice number, invoice date, due date, purchase order no. and the purchase details and put them on a single table (values on the table can be repeated per row)"
        st.experimental_rerun()

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["pdf", "jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            pdf_data = uploaded_file.read()
            images_data = convert_pdf_to_images(pdf_data)
            if images_data:
                image = Image.open(BytesIO(images_data[0]))
                st.image(image, caption="Uploaded PDF image", use_column_width=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

    prompt = load_prompt()

    if st.button("Send"):
        with st.spinner("Start processing..."):
            response = generate_response_llm(input_question=user_question, image=image, prompt=prompt)
            st.subheader("Response:")
            st.write(response)

if __name__ == "__main__":
    main()
