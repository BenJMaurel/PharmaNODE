# hash:sha256:10f2801cdbe410138c581a82ecfe726553b1fe00e9818dddcd68b9c544113c8b
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libdbus-1-3=1.12.20-2ubuntu4.1 \
        libegl1=1.4.0-1 \
        libgl1=1.4.0-1 \
        libnspr4=2:4.35-0ubuntu0.22.04.1 \
        libnss3=2:3.98-0ubuntu0.22.04.3 \
        libopengl0=1.4.0-1 \
        libx11-xcb1=2:1.7.5-1ubuntu0.3 \
        libxcb-glx0=1.14-3ubuntu3 \
        libxcb-icccm4=0.4.1-1.1build2 \
        libxcb-image0=0.4.0-2 \
        libxcb-keysyms1=0.4.0-1build3 \
        libxcb-randr0=1.14-3ubuntu3 \
        libxcb-render-util0=0.3.9-1build3 \
        libxcb-shape0=1.14-3ubuntu3 \
        libxcb-shm0=1.14-3ubuntu3 \
        libxcb-util1=0.4.0-1build2 \
        libxcb-xinerama0=1.14-3ubuntu3 \
        libxi6 \
        libxkbcommon-x11-0=1.4.0-1 \
        libxkbcommon0=1.4.0-1 \
        libxkbfile1 \
        shared-mime-info=2.1-2 \
        xdg-utils=1.1.3-4.1ubuntu3~22.04.1 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    chardet==7.4.3 \
    dash==4.1.0 \
    matplotlib==3.10.9 \
    numpy==2.4.4 \
    pandas==3.0.2 \
    plotly==6.7.0 \
    scikit-learn==1.8.0 \
    scipy==1.17.1 \
    seaborn==0.13.2 \
    tensorboard==2.20.0 \
    torch==2.11.0 \
    torchdiffeq==0.2.5 \
    torchvision==0.26.0

RUN Rscript -e 'remotes::install_version("argparse", "2.3.1")' \
    && Rscript -e 'remotes::install_version("tidyverse", "2.0.0")' \
    && Rscript -e 'remotes::install_version("mrgsolve", "1.7.2")' \
    && Rscript -e 'remotes::install_version("mapbayr", "0.10.2")' \
    && Rscript -e 'remotes::install_version("MESS", "0.6.0")' \
    && Rscript -e 'remotes::install_version("glue", "1.8.1")' \
    && Rscript -e 'remotes::install_version("furrr", "0.4.0")' \
    && Rscript -e 'remotes::install_version("RJSONIO", "2.0.0")' \
    && Rscript -e 'remotes::install_version("gridExtra", "2.3")'

