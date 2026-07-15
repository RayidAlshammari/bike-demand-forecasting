# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# 2. Set environment variables for optimized Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Copy the incredibly fast `uv` binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy requirements and install them globally using uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 6. Copy only the necessary artifacts for serving the API
COPY models/ ./models/
COPY src/ ./src/

# 7. Security: Create a non-root user and switch to it
RUN useradd -m apiuser
USER apiuser

# 8. Expose the port that FastAPI uses
EXPOSE 8000

# 9. Define the command to run the application using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]