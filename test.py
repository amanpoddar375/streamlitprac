import streamlit as st
from textblob import TextBlob
import spacy
from spacy import displacy

# Function to perform text classification using TextBlob
def text_classification(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Function to perform named entity recognition (NER) using SpaCy
def ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc

# Function to perform language translation using TextBlob
def translate(text, target_language):
    blob = TextBlob(text)
    translated_text = blob.translate(to=target_language)
    return translated_text

# Main function to define the Streamlit app
def main():
    st.title("Natural Language Processing (NLP) App")

    # Input text area
    input_text = st.text_area("Enter your text here:")

    # Selectbox for choosing NLP task
    nlp_task = st.selectbox("Select NLP Task:", ["Text Classification", "Named Entity Recognition (NER)", "Language Translation"])

    # Perform selected NLP task based on user input
    if st.button("Process"):
        if nlp_task == "Text Classification":
            sentiment = text_classification(input_text)
            st.write("Sentiment Analysis Results:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")
        elif nlp_task == "Named Entity Recognition (NER)":
            doc = ner(input_text)
            st.write("Named Entity Recognition (NER) Results:")
            displacy.render(doc, style="ent", jupyter=False)
        elif nlp_task == "Language Translation":
            target_language = st.text_input("Enter target language (e.g., 'fr' for French):")
            if target_language:
                translated_text = translate(input_text, target_language)
                st.write("Translation Results:")
                st.write(translated_text)

    # File uploader for uploading text files
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])

    # Process uploaded file
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        st.text_area("File Contents:", file_contents)
        if st.button("Process File"):
            if nlp_task == "Text Classification":
                sentiment = text_classification(file_contents)
                st.write("Sentiment Analysis Results:")
                st.write(f"Polarity: {sentiment.polarity}")
                st.write(f"Subjectivity: {sentiment.subjectivity}")
            elif nlp_task == "Named Entity Recognition (NER)":
                doc = ner(file_contents)
                st.write("Named Entity Recognition (NER) Results:")
                displacy.render(doc, style="ent", jupyter=False)
            elif nlp_task == "Language Translation":
                target_language = st.text_input("Enter target language (e.g., 'fr' for French):")
                if target_language:
                    translated_text = translate(file_contents, target_language)
                    st.write("Translation Results:")
                    st.write(translated_text)

if __name__ == "__main__":
    main()
