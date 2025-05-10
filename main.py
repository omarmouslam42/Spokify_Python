import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
import google.generativeai as genai
import os
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import re

load_dotenv()

app = FastAPI()

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUoABlR_TAkDcyrjCWKvIxJFAZWpBF_1I"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) 

# Gemini Flash for transcription and NLP tasks
generation_config = {
    "temperature": 0.1,
    "top_p": 0,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Embedding and chat models for RAG
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Vector DB directory
VECTOR_DB_PATH = "transcribed_audio_chunks_chroma"

# Temporary storage for transcriptions
transcription_cache = {}

# Function to transcribe audio
async def transcribe_audio_file(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_mime_types = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/opus"]

    if file.content_type not in allowed_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported types: wav, mp3, ogg, webm, opus")

    try:
        print(f"Received file: {file.filename}, type: {file.content_type}")  

        audio_bytes = await file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        print(f"Audio file encoded. Size (base64): {len(encoded_audio)} characters")
        response = model.generate_content([ 
            "Transcribe the following audio to text:",
            {
                "mime_type": file.content_type,
                "data": encoded_audio,
            },
        ]) 

        print("Transcription completed.") 

        return response.text
    except Exception as e:
        print(f"Error during transcription: {e}") 
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

def get_vectorstore():
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
    )

retriever = get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 5})

async def store_to_chroma(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"Storing {len(documents)} chunks")
    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print("Chunks stored and persisted.")

    return len(documents)

# NLP Features
async def summarize_text(text: str):
    try:
        response = model.generate_content(f"Summarize this: {text}")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {e}")

async def enhance_text(text: str):
    try:
        response = model.generate_content(
            f"Improve the following text by correcting grammar, enhancing clarity, and making it more natural-sounding.\n\n{text}"
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing text: {e}")

async def detect_topics(text: str):
    try:
        response = model.generate_content(f"List only the main topics discussed in this text as bullet points. No explanations:\n{text}")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting topics: {e}")
    
@app.post("/transcribe/") 
async def transcribe(file: UploadFile = File(...)):
    try:
        transcription = await transcribe_audio_file(file)

        language_response = model.generate_content(f"Identify the language of the following text. Respond only with the language name \n\n{transcription}")
        language = language_response.text.strip()

        main_point_response = model.generate_content(f"What is the main point of this text? Do not write any intro.\n\n{transcription}")
        main_point = main_point_response.text.strip()

        tags_response = model.generate_content(f"Extract at most 5 significant keywords from the following text. Just provide as Python list: [tag1, tag2, ...]\n\n{transcription}")
        raw_tags = tags_response.text.strip().replace("\n", "").replace("-", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]

        transcription_cache["latest"] = {
            "text": transcription,
            "metadata": {
                "language": language,
                "main_point": main_point,
                "tags": tags,
                "filename": file.filename,
                "upload_date": datetime.now().isoformat()
            }
        }

        stored_chunks = await store_to_chroma(transcription)

        return {
            "transcription": transcription,
            "metadata": transcription_cache["latest"]["metadata"],
            "message": f"Stored {stored_chunks} chunks to Chroma vector store."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

@app.get("/summarize/") 
async def summarize_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")

    summary = await summarize_text(transcription_cache["latest"]["text"])
    return {"summary": summary}

@app.get("/enhance/") 
async def enhance_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")

    enhanced_text = await enhance_text(transcription_cache["latest"]["text"])
    return {"enhanced_text": enhanced_text}

@app.get("/detect_topics/") 
async def detect_topics_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")

    topics = await detect_topics(transcription_cache["latest"]["text"])
    return {"topics": topics}

@app.get("/extract_tasks/") 
async def extract_tasks():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")

    prompt = f"""
    Extract all actionable tasks from this text:
    {transcription_cache['latest']['text']}

    Return only a list of tasks, each on a new line.
    """
    response = model.generate_content(prompt)
    return {"tasks": response.text.strip()}

@app.post("/ask_question/")
async def ask_question(question: str):
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_model,
        )

        raw_docs = vectorstore.get()["documents"]

        print("Total stored docs:", len(raw_docs))

        keywords = re.findall(r'\w+', question.lower())

        matching_docs = []
        for doc in raw_docs:
            doc_lower = doc.lower()
            if all(word in doc_lower for word in keywords):
                matching_docs.append(doc)

        if not matching_docs:
            return {"answer": "No related_paragraphs .", "related_paragraphs": ""}

        context = "\n\n".join([f"PARAGRAPHS{i+1}:\n{doc}" for i, doc in enumerate(matching_docs)])

        prompt = f"""
        form pragraph answer plesss
        {context}

        question: {question}
        answer:        """

        response = model.generate_content(prompt)

        return {
            "answer": response.text.strip(),
            "related_paragraphs": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e}")
