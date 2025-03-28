import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator

# Streamlit title
st.title("Medical AI Chatbot Assistant")
st.image('chatbot.jpg')

language_options = {"English": "en", "Telugu": "te", "Tamil": "ta"}
selected_language = st.selectbox("Select your preferred language for answers:", list(language_options.keys()))
lan=language_options[selected_language]

translator = GoogleTranslator()
folder_path = "db"


cached_llm = Ollama(model="wizardlm2:latest")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a medical assistant. Answer with reliable medical advice if known, otherwise respond 'I'm not sure about this'. [/INST] </s>
    [INST] {input} Context: {context} Answer: [/INST]"""
)

def retrieve_answer(query):
    st.write("Loading vector store...")
    try:
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None

    st.write("Creating chain...")
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.6}
        )
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        result = chain.invoke({"input": query})
        answer = result["answer"]

        if selected_language != "English":
            answer_translated = translator.translate(answer, dest=language_options[selected_language])
            answer = answer_translated  

        sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
        return answer,sources
    except Exception as e:
        st.error(f"Error creating chain: {e}")
        return None, None

st.header("Upload Medical Document")

# Allow user to upload a PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    save_file = f"pdf/{uploaded_file.name}"
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    chunks = text_splitters.split_documents(docs)
    
    st.write("Embedding medical document...")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=folder_path
        )
        vector_store.persist()
        st.success("Document processed and embedded for retrieval.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


st.header("Ask a Medical Question")

query = st.text_input("Enter your medical question here")
if selected_language != "English":
            query_translated = translator(source=lan, target='en').translate(query)
            query = query_translated
if st.button("Submit"):
    if query:

        answer,sources= retrieve_answer(query)
        

        if answer:
            st.write(f"**Answer in {selected_language}:**", answer)


            for source in sources:
                st.write(f"- Source: {source['source']}, Content: {source['page_content']}")
        else:
            st.warning("No answer found for this query. Please try rephrasing.")
    else:
        st.warning("Please enter a query.")



