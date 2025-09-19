FROM python:3.11-slim

WORKDIR /app

# Python requirements 먼저 복사
COPY requirements.txt .

# apt 패키지 설치 + Python 패키지 설치
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg build-essential libsndfile1 git && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# 앱 소스와 데이터 복사
COPY . .

# sample_data.json 호스트에 반드시 존재해야 함
# COPY sample_data.json /app/sample_data.json  # 이미 COPY . . 에 포함되므로 별도 필요 없음

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
