# FROM mcr.microsoft.com/devcontainers/base:focal
FROM movesrwth/stormpy:1.7.0
RUN apt-get update

RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev

#WORKDIR /home/vscode/
#RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -
#RUN echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list
#RUN apt-get update
#RUN apt-get -y install spot libspot-dev spot-doc python3-spot

#WORKDIR /home/vscode/
#RUN git clone https://github.com/moves-rwth/storm.git
#WORKDIR /home/vscode/storm
#RUN git checkout tags/1.7.0
#RUN mkdir build
#RUN pwd
#RUN ls
#WORKDIR /home/vscode/storm/build
#RUN cmake ..
#RUN make

WORKDIR /home/vscode
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    maven \
    uuid-dev \
    virtualenv

#RUN git clone https://github.com/moves-rwth/pycarl.git
#WORKDIR /home/vscode/pycarl
#RUN git checkout tags/2.1.0
RUN apt-get -y install python3 python3-pip python3-dev
RUN apt-get -y install python-is-python3
#RUN python setup.py build_ext develop

WORKDIR /home/vscode/

#RUN git clone https://github.com/moves-rwth/stormpy.git
#WORKDIR /home/vscode/stormpy
#RUN git checkout tags/1.7.0
#RUN python setup.py build_ext --storm-dir /storm/build/ develop

#WORKDIR /workspaces/coolmc
WORKDIR /
COPY ../requirements.txt .
RUN pip install -r requirements.txt


#COPY common common
#COPY custom_openai_gyms custom_openai_gyms
#COPY openai_gym_training openai_gym_training
#COPY safe_gym_training safe_gym_training
#COPY unit_testing unit_testing
#COPY verify_rl_agent verify_rl_agent
#COPY control_ui.py .
#COPY taxi_abstraction.json .
#COPY cool_mc.py .
#COPY start_ui.sh .

#RUN chmod -R 777 /home/mycoolmc

#RUN apt-get install -y iproute2

ENTRYPOINT /bin/bash
