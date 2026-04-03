FROM movesrwth/stormpy:1.12.0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Project must live at this exact path — required by Storm/MLflow internals
WORKDIR /workspaces/coolmc

# Clone COOL-MC (includes server/ and all source files)
RUN git clone https://github.com/LAVA-LAB/COOL-MC.git .

# Install COOL-MC Python deps (includes fastapi, uvicorn, aiofiles, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Persistent directories (bind-mounted from host at runtime)
RUN mkdir -p logs mlruns prism_files_user

EXPOSE 8765
EXPOSE 5000

# Startup script: launches the MLflow UI (background) + FastAPI server (foreground).
RUN printf '#!/bin/bash\nmlflow ui --backend-store-uri "file:///workspaces/coolmc/mlruns" --host 0.0.0.0 --port 5000 &\nexec uvicorn server.server:app --host 0.0.0.0 --port 8765\n' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
