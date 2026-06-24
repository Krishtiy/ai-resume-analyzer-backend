# Use a lightweight Python environment
FROM python:3.10-slim

# Set the working directory inside the server
WORKDIR /app

# Copy your requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NEW: Download the spaCy model directly via spaCy's built-in command
RUN python -m spacy download en_core_web_sm

# Copy all your Python code into the server
COPY . .

# Hugging Face Spaces strictly requires apps to run on port 7860
EXPOSE 7860

# Boot up the Uvicorn server on the required port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]