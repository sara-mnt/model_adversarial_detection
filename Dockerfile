FROM python:3.9.13-slim-bullseye
ARG user
ARG password
EXPOSE 8080
ADD ./requirements.txt /
RUN pip install --upgrade --extra-index-url https://piplivetech:pip2018@distribution.livetech.site -r /requirements.txt
ARG GATEWAY
ENV GATEWAY=$GATEWAY
ADD . /plugin
ENV PYTHONPATH=$PYTHONPATH:/plugin
WORKDIR /plugin/services
CMD python services.py