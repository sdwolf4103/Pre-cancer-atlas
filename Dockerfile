# 使用輕量的 Python 基底
FROM python:3.12-slim

# 建立非 root 使用者（更安全）
RUN useradd -m appuser

# 設定工作目錄
WORKDIR /app

# 先複製 requirements 並安裝（利用快取）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 明確複製程式碼 + 靜態資源（保證包含）
COPY precancer.py /app/precancer.py
COPY templates/ /app/templates/
COPY static/ /app/static/

# （可留著一次來看）在 build 階段列出內容，若 static 缺失會一眼看出
RUN echo "== BUILD LIST ==" && ls -la /app && echo "== /app/static ==" && ls -la /app/static || true

# 環境
ENV PORT=8080
EXPOSE 8080

# 以非 root 執行
USER appuser

# 用 gunicorn 啟動： <模組>:<app物件>
CMD exec gunicorn -w 2 -k gthread -t 120 -b :$PORT precancer:app