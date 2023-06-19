#!/bin/sh

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
sudo rm -rf "$CURR_DIR"/*/.ipynb_checkpoints/

sudo rm -rf "$CURR_DIR"/*/*/.pytest_cache/
sudo rm -rf "$CURR_DIR"/*/*/__pycache__/
sudo rm -rf "$CURR_DIR"/*/*/.ipynb_checkpoints/
