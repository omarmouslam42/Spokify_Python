import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
import google.generativeai as genai
import os
from pydantic import BaseModel
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUoABlR_TAkDcyrjCWKvIxJFAZWpBF_1I"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) 

# Gemini Flash model for transcription and NLP
generation_config = {
    "temperature": 0.1,
    "top_p": 0,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}
#configuration the model 
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Vector DB
VECTOR_DB_PATH = "transcribed_audio_chunks_chroma"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
transcription_cache = {}
async def transcribe_audio_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_mime_types = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/opus"]
    if file.content_type not in allowed_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        audio_bytes = await file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        response = model.generate_content([
            "Transcribe the following audio to text:",
            {
                "mime_type": file.content_type,
                "data": encoded_audio,
            },
        ])

        return {"transcription": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
# Store to Chroma
async def store_to_chroma(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    return len(documents)

def get_vectorstore():
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
    )

retriever = get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 5})

# NLP utilities
async def summarize_text(text: str):
    try:
        response = model.generate_content(f"Summarize this: {text}")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

@app.get("/enhance/")
async def enhance_latest():
    """Enhances the grammar and clarity of the latest transcribed text."""
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found. Please transcribe an audio file first.")

    try:
        enhanced_text = await enhance_text(transcription_cache["latest"]["text"])
        return {
            "enhanced_text": enhanced_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing text: {e}")
async def enhance_text(text: str):
    """Function to improve grammar and clarity of transcribed text."""
    try:
        response = model.generate_content(f"Improve the following text by correcting grammar, enhancing clarity, and making it more natural-sounding. "
            "Do not add new content or remove important meaning. Just return the improved version without any comments, notes, or explanation:\n\n"
            f"{text}")
        return response.text  # Extract enhanced text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing text: {e}")
@app.get("/detect_topics/")
async def detect_topics_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")
    
    try:
        topics = await detect_topics(transcription_cache["latest"]["text"])
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting topics: {e}")
async def detect_topics(text: str):
    response = model.generate_content(
        f"List only the main topics discussed in this text as bullet points. No explanations:\n{text}"

    )
    return response.text
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        transcription_result = await transcribe_audio_file(file)
        transcription_text = transcription_result["transcription"]

        language_response = model.generate_content(f"Identify the language:\n\n{transcription_text}")
        language = language_response.text.strip()

        main_point_response = model.generate_content(f"What is the main point?\n\n{transcription_text}")
        main_point = main_point_response.text.strip()

        tags_response = model.generate_content(f"Extract at most 5 keywords as Python list:\n\n{transcription_text}")
        raw_tags = tags_response.text.strip().replace("\n", "").replace("-", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]

        transcription_cache["latest"] = {
            "text": transcription_text,
            "metadata": {
                "language": language,
                "main_point": main_point,
                "tags": tags,
                "filename": file.filename,
                "upload_date": datetime.now().isoformat()
            }
        }

        chunks_stored = await store_to_chroma(transcription_text)

        return {
            "transcription": transcription_text,
            "metadata": transcription_cache["latest"]["metadata"],
            "chunks_stored": chunks_stored
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

@app.get("/summarize/")
async def summarize_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcription found.")

    summary = await summarize_text(transcription_cache["latest"]["text"])
    return {"summary": summary}

@app.get("/enhance/")
async def enhance_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcription found.")

    enhanced = await enhance_text(transcription_cache["latest"]["text"])
    return {"enhanced_text": enhanced}

@app.get("/detect_topics/")
async def detect_topics_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcription found.")

    topics = await detect_topics(transcription_cache["latest"]["text"])
    return {"topics": topics}

@app.get("/extract_tasks/")
async def extract_tasks():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcription found.")

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
        vectorstore = get_vectorstore()
        raw_docs = vectorstore.get()["documents"]
        keywords = re.findall(r'\w+', question.lower())

        matching_docs = []
        for doc in raw_docs:
            doc_lower = doc.lower()
            if all(word in doc_lower for word in keywords):
                matching_docs.append(doc)

        if not matching_docs:
            return {"answer": "No related paragraphs found.", "related_paragraphs": ""}

        context = "\n\n".join([f"PARAGRAPH {i+1}:\n{doc}" for i, doc in enumerate(matching_docs)])
        prompt = f"""
        Answer the question using the paragraphs below:
        {context}

        Question: {question}
        Answer:
        """

        response = model.generate_content(prompt)

        return {
            "answer": response.text.strip(),
            "related_paragraphs": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question error: {e}")