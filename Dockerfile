FROM python:3.11 as base

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install python dependenies
WORKDIR /app/ocean-clean
COPY ./requirements.txt /app/ocean-clean/requirements.txt
#RUN pip install torch-cpu
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/openai/CLIP.git

# Copy code
COPY ocean_clean /app/ocean-clean/ocean_clean
COPY scripts /app/ocean-clean/scripts
COPY data /app/ocean-clean/data

ENTRYPOINT ["/app/ocean-clean/scripts/start.sh"]

