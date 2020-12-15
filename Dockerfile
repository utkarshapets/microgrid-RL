FROM "tensorflow/tensorflow:1.15.4-gpu-py3"

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY ./gym-socialgame/ ./gym-socialgame/
RUN pip install -e ./gym-socialgame/ 

COPY ./rl_algos/ ./rl_algos/
RUN pip install -e ./rl_algos/stableBaselines/

WORKDIR ./tc/
