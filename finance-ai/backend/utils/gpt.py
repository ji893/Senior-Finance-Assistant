import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# 비동기 클라이언트를 사용합니다.
client = OpenAI(api_key=api_key)

async def get_gpt_response(prompt):
    """
    GPT-4에 프롬프트를 보내 비동기적으로 응답을 받습니다.
    """
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
