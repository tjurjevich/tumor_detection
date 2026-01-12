# import parent image
FROM python:3.10-slim 

# set container working directory
WORKDIR /app 

# copy contents into container
COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# port exposure
EXPOSE 8050

# run the Dash app
CMD ["python3", "-m", "frontend.app"]