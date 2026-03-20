FROM movesrwth/stormpy:1.7.0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Project must live at this exact path — required by Storm/MLflow internals
WORKDIR /workspaces/coolmc

# Clone COOL-MC (includes server/ and all source files)
RUN git clone https://github.com/LAVA-LAB/COOL-MC.git .

# Install COOL-MC Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Install server deps
RUN pip install --no-cache-dir fastapi>=0.110.0 uvicorn>=0.29.0 aiofiles>=23.0.0 python-multipart>=0.0.9

# Persistent directories (bind-mounted from host at runtime)
RUN mkdir -p logs mlruns prism_files_user

EXPOSE 8765

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8765"]
