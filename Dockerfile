FROM ubuntu:17.10

# Install
RUN apt-get -qq update
RUN apt-get install -y cmake g++
