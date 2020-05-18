# # local testing

# FROM continuumio/miniconda3:4.6.14
# LABEL maintainer="Billy Ermlick <wermlick@gmu.edu>"
# RUN apt-get update
# RUN apt-get install -y python-dev  python3-dev  gcc
# COPY ./ ./app
# RUN conda install --yes --freeze-installed \
#     geopandas==0.6.1  \
#     && conda clean -afy
# RUN pip install  --no-cache-dir -r ./app/requirements.txt
# COPY ./ ./app
# WORKDIR ./app/src
# RUN pip list
# EXPOSE 5000
# CMD ["python", "server.py"]




# dedicated web server https://medium.com/@gabimelo/developing-a-flask-api-in-a-docker-container-with-uwsgi-and-nginx-e089e43ed90e
FROM continuumio/miniconda3:4.6.14
LABEL maintainer="Billy Ermlick <wermlick@gmu.edu>"

RUN apt-get clean \
    && apt-get -y update
RUN apt-get install -y python-dev \
    && apt-get -y install python3-dev \
    && apt-get -y install  gcc build-essential uwsgi
RUN apt-get install -y --no-install-recommends \
        nginx supervisor


COPY ./requirements.txt /project/requirements.txt
RUN conda install --yes --freeze-installed \
    geopandas==0.6.1 \  
    && conda clean -afy
RUN conda install -c conda-forge uwsgi
RUN conda install -c conda-forge libiconv

#low ram install
RUN pip install --no-cache-dir  -r /project/requirements.txt


RUN useradd --no-create-home nginx
RUN rm /etc/nginx/sites-enabled/default
COPY server-conf/nginx.conf /etc/nginx/
COPY server-conf/flask-site-nginx.conf /etc/nginx/conf.d/
COPY server-conf/uwsgi.ini /etc/uwsgi/
COPY server-conf/supervisord.conf /etc/

COPY ./src /project

RUN chown -R nginx: /project/
RUN chmod -R u+w /project/

WORKDIR /project
RUN pip list
RUN ls
EXPOSE 80
CMD ["/usr/bin/supervisord"]