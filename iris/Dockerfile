# Specifies base image and tag
FROM learners:latest

WORKDIR /home

RUN apt -y update
RUN apt -y upgrade

# Installs additional packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install -U google-cloud
RUN pip3 install -U google-cloud.aiplatform
RUN pip3 install -U gcsfs

# Temporarily downgrade `protobuf` to avoid <https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly>
RUN pip3 install protobuf==3.20.*  

# Copies the trainer code to the docker image.
COPY *.py *.md *.json /home/

ENV INSIDE_GCP Yes

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "learner.py"]
