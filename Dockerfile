# Use a slim Python 3.12 image
FROM python:3.12-slim

# Install any system dependencies (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app's code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app_main_hf.py", "--server.enableCORS", "false"]
