FROM ubuntu:17.04

# Install
RUN apt-get -qq update
RUN apt-get install -y cmake g++
