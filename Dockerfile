FROM bitnami/spark:latest

USER root

# Installer Python et pip
RUN install_packages python3 python3-pip

# Copier et installer les dépendances
COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && pip3 install -r /tmp/requirements.txt

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

