# Use a slim Python 3.12 image
FROM python:3.12-slim

# Copy your packages.txt file into the container
COPY packages.txt .

# Update package lists, upgrade installed packages, then install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    xargs -a packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Check Installed Packages
RUN pip list

# Copy the rest of your app's code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app_main_hf.py", "--server.enableCORS", "false"]
