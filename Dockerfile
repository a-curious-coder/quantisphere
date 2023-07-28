FROM python:3.10.9

# Add requirements.txt and src folder to the image
ADD requirements.txt .
RUN pip install -r requirements.txt

# Set the DISPLAY environment variable
ENV DISPLAY=:0

WORKDIR /app

# Add all src files to the image
ADD src .

CMD [ "python", "data_collector.py" ]
# CMD ["ls"]
