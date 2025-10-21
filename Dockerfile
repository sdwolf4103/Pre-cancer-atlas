# 使用輕量的 Python 基底
FROM python:3.12-slim

# 建立非 root 使用者（更安全）
RUN useradd -m appuser

# 設定工作目錄
WORKDIR /app

# 先複製 requirements 並安裝（利用快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再複製其他程式碼
COPY . .

# Cloud Run 會給我們 $PORT，Flask/gunicorn 要綁定它
ENV PORT=8080

# 開放容器內的 8080
EXPOSE 8080

# 用 gunicorn 啟動： <模組>:<app物件>
# 這裡是 precancer.py 內的 app 物件 → precancer:app
USER appuser
CMD exec gunicorn -w 2 -k gthread -t 120 -b :$PORT precancer:app