# Specifies base image and tag
FROM learners:latest

# Declare the working directory.
WORKDIR /home

# Set environment variables.
ENV INSIDE_GCP Yes

# Update Linux.
RUN apt -y update
RUN apt -y upgrade

# Installs additional packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install protobuf==3.20.*  # Temporarily downgrade `protobuf` to avoid <https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly>

# Copies the trainer code to the docker image.
COPY *.py *.md cloud_params.json /home/

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "learner.py"]
