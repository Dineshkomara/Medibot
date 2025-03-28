import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr

# Initialize Translator for multilingual support
translator = GoogleTranslator()

# Streamlit title
st.title("Medical AI Chatbot Assistant")
# st.image('chatbot.jpg')
st.image('chatbot.jpg')


# Define folder path for vector store and media folders
folder_path = "db"
media_base_folder = "medibot"  # Folder containing the media directories like "bleeding", "cpr", etc.

# Initialize models and embeddings
cached_llm = Ollama(model="deepseek-llm:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

# Define the prompt with a medical context disclaimer
raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a medical assistant. Provide a brief, single-paragraph response. Keep your answer concise and focused on the main points. If unsure, respond 'I'm not sure about this'. [/INST] </s>
    [INST] {input} Context: {context} Answer: [/INST]"""
)

# Function to retrieve the answer using vector store
def retrieve_answer(query):
    st.write("Loading vector store...")
    try:
        # Load vector store for retrieval
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None
    
    try:
        # Create retriever and document processing chain
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.7}
        )
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        # Retrieve response with user's query
        result = chain.invoke({"input": query})

        # Extract sources for display
        sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
        return result["answer"], sources
    except Exception as e:
        st.error(f"Error creating chain: {e}")
        return None, None

# Function for speech-to-text conversion
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Function to get matching folder based on the query
def get_best_matching_folder(query):
    # You can implement a more advanced matching here, for now it checks if the query matches any folder name
    folders = os.listdir(media_base_folder)
    matching_folder = None
    max_score = 0

    for folder in folders:
        score = 0
        if folder.lower() in query.lower():
            score += 1
        # You can add more sophisticated matching logic here using NLP techniques like sentence transformers
        
        if score > max_score:
            max_score = score
            matching_folder = folder

    return os.path.join(media_base_folder, matching_folder) if matching_folder else None

# Function to get media files from a folder
def get_media_from_folder(folder_path):
    # Get all image files (JPG, PNG, GIF)
    media_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return media_files

# Function to return media based on query
def return_media(query):
    matching_folder = get_best_matching_folder(query)
    if matching_folder:
        media_files = get_media_from_folder(matching_folder)
        return media_files
    else:
        return []

# Streamlit layout for uploading PDF document
st.header("Upload Medical Document")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    save_file = f"pdf/{uploaded_file.name}"
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    # Load PDF document for processing
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    chunks = text_splitters.split_documents(docs)
    
    st.write("Embedding medical document...")
    try:
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=folder_path
        )
        vector_store.persist()
        st.success("Document processed and embedded for retrieval.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

language_options = {"English": "en", "Telugu": "te", "Tamil": "ta"}
selected_language = st.selectbox("Select your preferred language for answers:", list(language_options.keys()))
lan=language_options[selected_language]

# Query section for chatbot interaction
st.header("Ask a Medical Question")

query = st.text_input("Enter your medical question here")
st.write("OR")
audio_bytes = audio_recorder(
    text="Record your question",
    recording_color="#ff4d4d",
    neutral_color="#6699ff",
    icon_name="microphone-alt",
    icon_size="5x"
)
if not query:
    if audio_bytes:
        # Save recorded audio
        audio_file = "audio_query.wav"
        with open(audio_file, "wb") as f:
            f.write(audio_bytes)

        st.success("Audio recorded successfully.")
        query = speech_to_text(audio_file)
if query:
    language = selected_language
    if language=="Telugu":
        query=GoogleTranslator(source="te", target="en").translate(query)
    elif language=="Tamil":
        query=GoogleTranslator(source="ta", target="en").translate(query)
    st.write(query)
    if st.button("Submit"):
        # Retrieve answer and sources
        answer, sources = retrieve_answer(query)
        
        # Display answer and sources
        if answer:
            translated_answer = None
            if language == "Telugu":
                try:
                    translated_answer = GoogleTranslator(source="auto", target="te").translate(answer)
                except Exception as e:
                    st.error("Error translating answer to Telugu.")
            elif language == "Tamil":
                try:
                    translated_answer = GoogleTranslator(source="auto", target="ta").translate(answer)
                except Exception as e:
                    st.error("Error translating answer to Tamil.")
            else:
                translated_answer = answer
            
            st.write(f"**Answer in {language}:**", translated_answer)
            
            # Return the corresponding images/GIFs
            media = return_media(query)
            st.write("Relevant Media:")
            if media:
                for item in media:
                    st.image(os.path.join(get_best_matching_folder(query), item), use_container_width=True)
            else:
                st.write("No media found for this query.")
        else:
            st.warning("No answer found for this query. Please try rephrasing.")
else:
    st.warning("Please enter or record a query.")
