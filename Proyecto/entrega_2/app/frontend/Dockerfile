# app/frontend/dockerfile

FROM python:3.10-slim
	
WORKDIR /app
	
COPY requirements.txt .
	
RUN pip install --no-cache-dir -r requirements.txt
	
COPY . .
	
ENV BACKEND_URL=http://backend:8000
	
EXPOSE 7860
	
CMD ["python", "app.py"]
 