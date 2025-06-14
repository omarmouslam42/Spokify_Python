import os
import re
import base64
from typing import List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Body, Form, HTTPException
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from trello import TrelloClient
import requests
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ✅ تحميل المتغيرات من .env
load_dotenv()

# ✅ تهيئة FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ مفاتيح Gemini و Trello
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("api_key")  # Trello public key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ كاش مؤقت للنصوص
transcription_cache = {}

# ✅ إعداد نموذج Gemini
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

class BoardRequest(BaseModel):
    board_name: str

class CardRequest(BaseModel):
    board_name: str
    list_name: str
    card_name: str
    card_desc: str = "" 
    due_date: str = None
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # حجم chunk أكبر لتوفير سياق أفضل
        chunk_overlap=100      # تداخل عشان السياق يكمل بعضه
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    return len(documents)

# Load vector store
def get_vectorstore():
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
    )

# استخدم MMR search بدل Similarity
retriever = get_vectorstore().as_retriever(
    search_type="mmr",  # MMR للنتائج الأكثر تنوعًا ودلالة
    search_kwargs={"k": 5}
)

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

        # ✅ إعادة تحميل الـ Retriever بعد التخزين
        global retriever
        chunks_stored = await store_to_chroma(transcription_text)
        retriever = get_vectorstore().as_retriever(search_type="mmr", search_kwargs={"k": 5})

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
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_question/")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()

        # البحث في المتجهات
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        docs = retriever.invoke(question)

        if not docs:
            return {"answer": "No relevant paragraphs found.", "related_paragraphs": ""}

        # بناء السياق من الفقرات
        context = "\n\n".join([f"PARAGRAPH {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

        # بناء البرومبت
        prompt = f"""
        Use the following paragraphs to answer the question.

        {context}

        Question: {question}
        Answer:
        """

        # إرسال البرومبت إلى ChatGoogleGenerativeAI
        response = chat_model.invoke(prompt)

        return {
            "answer": response.content.strip(),
            "related_paragraphs": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question error: {e}")
# @app.post("/extract_and_add_tasks/")
# async def extract_and_add_tasks(
#     board_name: str = Body(...),
#     list_name: str = Body(...),
#     members: List[str] = Body(default=[]),  # قائمة الإيميلات
# ):
#     if "latest" not in transcription_cache:
#         raise HTTPException(status_code=400, detail="No transcribed text found.")

#     try:
#         # 🧠 1. استخراج المهام من النص
#         prompt = f"""
#         Extract all actionable tasks from this text:
#         {transcription_cache["latest"]["text"]}
#         Return only a list of tasks, each on a new line, without extra explanations.
#         """
#         response = model.generate_content(prompt)
#         tasks_text = response.text.strip()
#         tasks = [task.strip("-• ") for task in tasks_text.split("\n") if task.strip()]

#         if not tasks:
#             raise HTTPException(status_code=400, detail="No tasks found.")

#         # 📦 2. إيجاد أو إنشاء البورد
#         boards = client.list_boards()
#         board = next((b for b in boards if b.name.lower() == board_name.lower()), None)
#         if not board:
#             board = client.add_board(board_name)
#         board_id = board.id

#         # ➕ 3. إضافة الأعضاء إلى البورد
#         for member_email in members:
#             add_member_url = f"https://api.trello.com/1/boards/{board_id}/members"
#             r = requests.put(
#                 add_member_url,
#                 params={
#                     "key": api_key,
#                     "token": api_token,
#                     "email": member_email,
#                     "type": "normal"
#                 }
#             )
#             if r.status_code not in [200, 202]:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Failed to add member {member_email}: {r.text}"
#                 )

#         # 🧾 4. إيجاد أو إنشاء القائمة (List)
#         lists = board.list_lists()
#         target_list = next((l for l in lists if l.name.lower() == list_name.lower()), None)
#         if not target_list:
#             target_list = board.add_list(name=list_name, pos="bottom")

#         # 📌 5. إضافة المهام كبطاقات
#         added_cards = []
#         for task in tasks:
#             card = target_list.add_card(name=task)
#             added_cards.append({
#                 "task": task,
#                 "card_id": card.id,
#                 "card_url": card.url
#             })

#         return {
#             "message": "Tasks extracted and added to Trello successfully.",
#             "tasks": added_cards,
#             "members_added": members
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during task extraction or Trello interaction: {e}")

# @app.post("/extract_and_add_tasks/")
# async def extract_and_add_tasks(
#     board_name: str = Body(...),
#     list_name: str = Body(...),
#     members: List[str] = Body(default=[]),
#     user_token: str = Body(...),
# ):
#     if "latest" not in transcription_cache:
#         raise HTTPException(status_code=400, detail="لم يتم العثور على نص لتحليله. يرجى تسجيل صوت أولاً.")

#     try:
#         # استخراج المهام باستخدام Gemini
#         prompt = f"""
#         Extract all actionable tasks from this text:
#         {transcription_cache["latest"]["text"]}
#         Return only a list of tasks, each on a new line, without extra explanations.
#         """
#         response = model.generate_content(prompt)
#         tasks_text = response.text.strip()
#         tasks = [task.strip("-• ") for task in tasks_text.split("\n") if task.strip()]

#         if not tasks:
#             raise HTTPException(status_code=400, detail="لم يتم العثور على أي مهام قابلة للتنفيذ.")

#         # إعداد TrelloClient باستخدام توكن المستخدم
#         client = TrelloClient(
#             api_key=api_key,
#             api_secret=None,
#             token=user_token
#         )

#         # الحصول على البورد أو إنشاؤه
#         boards = client.list_boards()
#         board = next((b for b in boards if b.name.lower() == board_name.lower()), None)
#         if not board:
#             board = client.add_board(board_name)

#         # الحصول على الليست أو إنشاؤها
#         lists = board.list_lists()
#         target_list = next((l for l in lists if l.name.lower() == list_name.lower()), None)
#         if not target_list:
#             target_list = board.add_list(name=list_name, pos="bottom")

#         # إضافة الكروت
#         added_cards = []
#         for task in tasks:
#             card = target_list.add_card(name=task)
#             added_cards.append({
#                 "task": task,
#                 "card_id": card.id,
#                 "card_url": card.url
#             })

#         # دعوة الأعضاء للبورد
#         board_id = board.id
#         for member in members:
#             add_member_url = f"https://api.trello.com/1/boards/{board_id}/members"
#             response = requests.put(
#                 add_member_url,
#                 params={
#                     "key": api_key,
#                     "token": user_token,
#                     "email": member,
#                     "type": "normal",
#                 },
#             )
#             if response.status_code not in [200, 202]:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"فشل في دعوة العضو {member}: {response.text}",
#                 )

#         return {
#             "message": "تم استخراج المهام وإضافتها بنجاح إلى Trello.",
#             "tasks": added_cards
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء التعامل مع Trello: {str(e)}")


import requests


@app.post("/extract_and_add_tasks/")
async def extract_and_add_tasks(
    board_name: str = Body(...),
    list_name: str = Body(...),
    members: List[str] = Body(default=[]),
    user_token: str = Body(...),
):
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="لم يتم العثور على نص لتحليله. يرجى تسجيل صوت أولاً.")

    try:
        # استخراج المهام باستخدام نموذج Gemini (أو بديل حسب مشروعك)
        prompt = f"""
        Extract all actionable tasks from this text:
        {transcription_cache["latest"]["text"]}
        Return only a list of tasks, each on a new line, without extra explanations.
        """
        # ← استبدل هذا بالسطر الخاص بنموذج الذكاء الاصطناعي اللي بتستخدمه
        response = model.generate_content(prompt)
        tasks_text = response.text.strip()
        tasks = [task.strip("-• ") for task in tasks_text.split("\n") if task.strip()]

        if not tasks:
            raise HTTPException(status_code=400, detail="لم يتم العثور على أي مهام قابلة للتنفيذ.")

        # إعداد Trello client
        client = TrelloClient(
            api_key=api_key,
            api_secret=None,
            token=user_token
        )

        # الحصول أو إنشاء البورد
        boards = client.list_boards()
        board = next((b for b in boards if b.name.lower() == board_name.lower()), None)
        if not board:
            board = client.add_board(board_name)

        # الحصول أو إنشاء الليست
        lists = board.list_lists()
        target_list = next((l for l in lists if l.name.lower() == list_name.lower()), None)
        if not target_list:
            target_list = board.add_list(name=list_name, pos="bottom")

        # إضافة الكروت
        added_cards = []
        for task in tasks:
            card = target_list.add_card(name=task)
            added_cards.append({
                "task": task,
                "card_id": card.id,
                "card_url": card.url
            })

        # ✅ إضافة الأعضاء إلى البورد
        board_id = board.id
        for member_email in members:
            invite_url = f"https://api.trello.com/1/boards/{board_id}/members"
            invite_params = {
                "key": api_key,
                "token": user_token,
                "email": member_email,
                "type": "normal"
            }
            response = requests.put(invite_url, params=invite_params)

            if response.status_code not in [200, 202]:
                raise HTTPException(
                    status_code=400,
                    detail=f"فشل في دعوة العضو {member_email}: {response.text}"
                )

        return {
            "message": "تم استخراج المهام وإضافتها إلى Trello بنجاح.",
            "tasks": added_cards
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء التعامل مع Trello: {str(e)}")

@app.get("/extract_title/")
async def extract_title():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcription found.")

    text = transcription_cache["latest"]["text"]

    prompt = (
        "Generate a short and clear title (maximum 5 words) for the following text. "
        "Do not add any formatting, stars, quotation marks, or explanations. Just return the title only:\n\n"
        f"{text}"
    )

    try:
        response = model.generate_content(prompt)
        raw_title = response.text.strip()

        # Remove unwanted formatting like *, ", newlines, etc.
        clean_title = re.sub(r'[*"\n]', '', raw_title).strip()

        # Optional: Enforce word limit strictly
        words = clean_title.split()
        if len(words) > 5:
            clean_title = ' '.join(words[:5])

        return {"title": clean_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting title: {e}")
