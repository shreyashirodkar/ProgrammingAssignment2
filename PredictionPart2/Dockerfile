FROM python:3.8

WORKDIR /usr/src/app

COPY . .
COPY /winemodel.model/ /model/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;
	
	# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;
	
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

CMD ["python","./winequalityprediction.py"]

