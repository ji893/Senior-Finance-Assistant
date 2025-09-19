import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import whisper
import imageio_ffmpeg
from dotenv import load_dotenv

from PIL import Image, ImageOps, ImageFilter
import pytesseract

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from pydub import AudioSegment

# ------------------------------
# Windows ffmpeg 환경 설정
# ------------------------------
ffmpeg_path = r"C:\Users\ji\Documents\finance-ai\ffmpeg-8.0-essentials_build\bin"
ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
if not os.path.exists(ffmpeg_exe):
    raise FileNotFoundError(f"ffmpeg.exe가 존재하지 않습니다: {ffmpeg_exe}")
os.environ["PATH"] += os.pathsep + ffmpeg_path
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
print("Whisper에서 사용할 ffmpeg 경로:", imageio_ffmpeg.get_ffmpeg_exe())

# ------------------------------
# 경로 설정
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data.json")
SERVER_IMAGE_PATH = os.path.join(BASE_DIR, "신용카드고지서.jpg")  # 서버 내 이미지 경로

app = FastAPI(title="Senior Finance AI Service", version="0.6.0")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------
# Whisper 모델 로드
# ------------------------------
model = whisper.load_model("medium")

# ------------------------------
# 유틸 함수
# ------------------------------
def convert_korean_number_to_digit(text: str) -> str:
    mapping = {"공": "0", "일": "1", "이": "2", "삼": "3", "사": "4",
               "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9"}
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def transcribe_audio(file_path: str) -> str:
    result = model.transcribe(file_path)
    return convert_korean_number_to_digit(result["text"].strip())

def load_sample_data():
    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_wav(file_path: str) -> str:
    """webm/mp3/m4a → wav 변환 (이미 wav면 그대로 리턴)"""
    if file_path.endswith(".wav"):
        return file_path
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_path, format="wav")
    return wav_path

# ------------------------------
# LLM + Embedding 준비
# ------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

# ------------------------------
# RAG + GPT 함수 (OCR 분리)
# ------------------------------
def rag_query_category(query: str, category: str, data: dict, ocr_text: str = ""):
    docs = []

    if category == "risk-alerts":
        for alert in data.get("risk_alerts", []):
            text = f"[{alert['type']}] {alert['message']}"
            docs.append(Document(page_content=text, metadata=alert))

    elif category == "spending-summary":
        for item in data.get("spending_history", []):
            text = f"{item['category']} {item['count']}회, 총 {item['amount']}원. ({item['description']})"
            docs.append(Document(page_content=text, metadata=item))
        if ocr_text:
            docs.append(Document(page_content=f"[OCR] {ocr_text}", metadata={"type": "ocr"}))

    elif category == "recommended-card":
        card = data.get("recommended_card", {})
        if card:
            text = f"추천카드: {card['name']} ({card['type']}), 이유: {card['reason']}"
            docs.append(Document(page_content=text, metadata=card))

    elif category == "fraud-detection":
        history_text = "\n".join([f"[HIST] {item['category']} {item['count']}회, 총 {item['amount']}원 ({item['description']})"
                                  for item in data.get("spending_history", [])])
        if history_text:
            docs.append(Document(page_content=history_text, metadata={"type": "history"}))

        for tx in data.get("recent_transactions", []):
            text = f"[NEW] {tx['date']} {tx['merchant']} {tx['category']} {tx['amount']}원"
            docs.append(Document(page_content=text, metadata=tx))

    if not docs:
        return "관련 데이터가 없습니다."

    vectorstore = FAISS.from_documents(docs, embeddings)
    retrieved = vectorstore.similarity_search(query, k=10)
    context = "\n\n".join([d.page_content for d in retrieved])

    # ------------------
    # GPT 프롬프트 작성
    # ------------------
    if category in ["spending-summary", "fraud-detection"]:
        prompt = f"""
        
        
        
당신은 은행 직원입니다. 고객의 최근 결제 내역과 OCR 고지서 데이터를 분석하세요.

1. 각 카테고리를 별도로 구분하고, 
   - **카테고리 이름**: 사용 횟수, 총 금액 (설명)
2. 의심 거래는 ⚠️ 아이콘과 빨간색 표시, 의심 거래 이야기는 한 번만
3. 금액은 쉼표 포함, 보기 쉽게 정렬
4. HTML에서 바로 줄바꿈이 적용되도록 <br> 사용
5. 마지막에 "궁금증이 해결되셨나요?"를 붙이세요.


데이터:
{context}

질문: {query}
답변:"""
    else:
        prompt = f"데이터:\n{context}\n\n질문: {query}\n답변:"

    response = llm.predict(prompt)
    return response

# ------------------------------
# 홈 페이지
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# ------------------------------
# STT + OCR → 페이지 이동
# ------------------------------
@app.post("/stt-gpt", response_class=HTMLResponse)
async def speech_to_text_gpt(
    request: Request,
    files: list[UploadFile] = File(...),
    include_image: str = Query("false")
):
    include_image = include_image.lower() == "true"
    temp_files = []
    transcript_texts = []
    ocr_text = ""

    try:
        # 업로드된 음성 파일 처리
        for file in files:
            ext = os.path.splitext(file.filename.lower())[1]
            temp_file = os.path.join(BASE_DIR, f"temp_upload{ext}")
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_file)

            if ext in [".wav", ".mp3", ".m4a", ".webm"]:
                wav_file = ensure_wav(temp_file)
                transcript_texts.append(transcribe_audio(wav_file))

        # OCR 처리 (include_image=True일 때만)
        if include_image and os.path.exists(SERVER_IMAGE_PATH):
            try:
                image = Image.open(SERVER_IMAGE_PATH)
                image = ImageOps.grayscale(image)
                image = image.filter(ImageFilter.MedianFilter())
                ocr_text = convert_korean_number_to_digit(
                    pytesseract.image_to_string(image, lang="kor+eng").strip()
                )
            except Exception as e:
                return HTMLResponse(f"이미지 처리 중 오류 발생: {str(e)}", status_code=400)

        # transcript: 음성 텍스트만
        transcript = "\n".join(transcript_texts)
        transcript_lower = transcript.lower()
        print("인식 텍스트:", transcript)

        # 키워드 기반 페이지 선택 (음성 텍스트 기준)
        selected_pages = []
        if "보이스피싱" in transcript_lower or "위험" in transcript_lower:
            selected_pages.append("risk-alerts")
        if "결제" in transcript_lower or "지출" in transcript_lower:
            selected_pages.append("spending-summary")
        if "카드" in transcript_lower or '추천' in transcript_lower:
            selected_pages.append("recommended-card")
        if "거래" in transcript_lower or "내역" in transcript_lower:
            selected_pages.append("fraud-detection")
        if '고지서' in transcript_lower or '뭐야' in transcript_lower:
            selected_pages.append("ocr")

        if not selected_pages:
            return HTMLResponse(f"인식된 텍스트: {transcript}<br>어떤 페이지로 갈지 알 수 없습니다.", status_code=200)

        # 단일 페이지 이동
        if len(selected_pages) == 1:
            page = selected_pages[0]
            return RedirectResponse(url=f"/{page}-auto?transcript={transcript}", status_code=303)

        # 다중 페이지
        pages_str = ",".join(selected_pages)
        return HTMLResponse(f"""
            <html>
            <body>
            <script>
                window.location.href = "/multi-info?pages={pages_str}&transcript={transcript}";
            </script>
            </body>
            </html>
        """)

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# ------------------------------
# 페이지 엔드포인트
# ------------------------------
@app.get("/risk-alerts-auto", response_class=HTMLResponse)
async def risk_alerts_auto(request: Request, transcript: str):
    sample_data = load_sample_data()
    gpt_answer = rag_query_category(transcript, "risk-alerts", sample_data)
    return templates.TemplateResponse(
        "risk_alerts.html",
        {"request": request, "transcript": transcript,
         "risk_alerts": sample_data.get("risk_alerts", []),
         "gpt_answer": gpt_answer}
    )

@app.get("/spending-summary-auto", response_class=HTMLResponse)
async def spending_summary_auto(request: Request, transcript: str):
    sample_data = load_sample_data()
    gpt_answer = rag_query_category(transcript, "spending-summary", sample_data)
    return templates.TemplateResponse(
        "spending_summary.html",
        {"request": request, "transcript": transcript,
         "spending_summary": sample_data.get("spending_history", []),
         "gpt_answer": gpt_answer}
    )

@app.get("/recommended-card-auto", response_class=HTMLResponse)
async def recommended_card_auto(request: Request, transcript: str):
    sample_data = load_sample_data()
    gpt_answer = rag_query_category(transcript, "recommended-card", sample_data)
    return templates.TemplateResponse(
        "recommended_card.html",
        {"request": request, "transcript": transcript,
         "recommended_card": sample_data.get("recommended_card", {"name":"N/A","type":"N/A","reason":""}),
         "gpt_answer": gpt_answer}
    )

@app.get("/fraud-detection-auto", response_class=HTMLResponse)
async def fraud_detection_auto(request: Request, transcript: str):
    sample_data = load_sample_data()
    gpt_answer = rag_query_category(transcript, "fraud-detection", sample_data)
    return templates.TemplateResponse(
        "fraud_detection.html",
        {
            "request": request,
            "transcript": transcript,
            "spending_history": sample_data.get("spending_history", []),
            "recent_transactions": sample_data.get("recent_transactions", []),
            "gpt_answer": gpt_answer
        }
    )

@app.get("/ocr-auto", response_class=HTMLResponse)
async def ocr_auto(request: Request, transcript: str):
    if not os.path.exists(SERVER_IMAGE_PATH):
        return HTMLResponse(f"서버 이미지가 없습니다: {SERVER_IMAGE_PATH}", status_code=400)
    try:
        image = Image.open(SERVER_IMAGE_PATH)
        image = ImageOps.grayscale(image)
        image = image.filter(ImageFilter.MedianFilter())
        ocr_text = convert_korean_number_to_digit(
            pytesseract.image_to_string(image, lang="kor+eng").strip()
        )
    except Exception as e:
        return HTMLResponse(f"OCR 처리 중 오류: {str(e)}", status_code=500)

    return templates.TemplateResponse(
        "ocr.html",
        {
            "request": request,
            "transcript": transcript,
            "ocr_text": ocr_text
        }
    )

@app.get("/multi-info", response_class=HTMLResponse)
async def multi_info(request: Request, pages: str, transcript: str):
    sample_data = load_sample_data()
    selected_pages = pages.split(",")

    answers = {}
    for page in selected_pages:
        # OCR 필요하면 ocr_text 전달
        ocr_text = ""
        if page == "ocr" and os.path.exists(SERVER_IMAGE_PATH):
            try:
                image = Image.open(SERVER_IMAGE_PATH)
                image = ImageOps.grayscale(image)
                image = image.filter(ImageFilter.MedianFilter())
                ocr_text = convert_korean_number_to_digit(
                    pytesseract.image_to_string(image, lang="kor+eng").strip()
                )
            except:
                ocr_text = ""
        answers[page] = rag_query_category(transcript, page, sample_data, ocr_text)

    return templates.TemplateResponse(
        "multi_info.html",
        {
            "request": request,
            "transcript": transcript,
            "selected_pages": selected_pages,
            "risk_alerts": sample_data.get("risk_alerts", []),
            "spending_summary": sample_data.get("spending_history", []),
            "recommended_card": sample_data.get("recommended_card", {"name":"N/A","type":"N/A","reason":""}),
            "recent_transactions": sample_data.get("recent_transactions", []),
            "answers": answers
        }
    )
