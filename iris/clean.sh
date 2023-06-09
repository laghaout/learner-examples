#!/bin/sh

PACKAGE=iris
CURR_DIR="."

sudo rm -rf "$CURR_DIR"/*~
sudo rm -rf "$CURR_DIR"/lesson/
sudo rm -rf "$CURR_DIR"/.pytest_cache/
sudo rm -rf "$CURR_DIR"/__pycache__/
sudo rm -rf "$CURR_DIR"/.ipynb_checkpoints/
sudo rm -rf "$CURR_DIR"/*/*~
sudo rm -rf "$CURR_DIR"/*/lesson/
sudo rm -rf "$CURR_DIR"/*/.pytest_cache/
sudo rm -rf "$CURR_DIR"/*/__pycache__/
sudo rm -rf "$CURR_DIR"/build/ "$CURR_DIR"/dist/ "$CURR_DIR"/$PACKAGE.egg-info/

sudo rm -rf "$CURR_DIR"/log.txt
sudo rm -rf "$CURR_DIR"/Data/
sudo rm -rf "$CURR_DIR"/exploration/
sudo rm -rf "$CURR_DIR"/dashboard/
