# Specifies base image and tag
FROM learners:latest

WORKDIR /home

# Installs additional packages
# RUN pip install 

# Copies the trainer code to the docker image.
COPY *.py *.md *.json /home/

ENV INSIDE_GCP Yes

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "learner.py"]
