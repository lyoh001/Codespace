FROM ubuntu:20.04

ENV TZ=Australia/Melbourne

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get install man-db -y \
    && apt-get install wget -y \
    && apt-get install curl -y \
    && apt-get install gnupg -y \
    && apt-get install software-properties-common -y \
    && apt-get install ca-certificates -y \
    && apt-get install apt-transport-https -y \
    && apt-get install lsb-release -y \
    && apt-get install zsh -y \
    && apt-get install tzdata -y \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install git -y \
    && apt-get install npm -y \
    && apt-get install python3.8 -y \
    && apt-get install python3-pip -y \
    && apt-get install python3-venv -y \
    && python3 -m pip install --upgrade pip \
    && sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

EXPOSE 7000