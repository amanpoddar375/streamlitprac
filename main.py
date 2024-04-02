import streamlit as st
import nltk
import spacy
from transformers import pipeline

# Load pre-trained models
classifier = pipeline("text-classification", model="bert-base-uncased")
ner_model = spacy.load("en_core_web_sm")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

def main():
    st.title("Natural Language Processing App")
    
    # text input or file upload
    text = st.text_area("Enter text or upload a file")
    
    # NLP task selection
    task = st.selectbox("Select an NLP task", ["Text Classification", "Named Entity Recognition", "Language Translation"])
    
    if task == "Text Classification":
        # Text classification
        if text:
            result = classifier(text)
            st.write(f"Classification: {result[0]['label']}")
    
    elif task == "Named Entity Recognition":
        # Named entity recognition
        if text:
            doc = ner_model(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.write("Named Entities:")
            for entity in entities:
                st.write(f"{entity[0]} - {entity[1]}")
    
    elif task == "Language Translation":
        # Language translation
        source_lang = st.selectbox("Source Language", ["English", "French"])
        target_lang = st.selectbox("Target Language", ["French", "English"])
        
        if text:
            translated = translator(text)[0]["translation_text"]
            st.write(f"Translated Text ({target_lang}):")
            st.write(translated)


if __name__ == "__main__":
    main()