#!/bin/bash

# run detect_web

# warn:docker inner path
export IMAGE_PATH="./inference/images"

WEB_PORT=8080 WEB_DEBUG=true SAVE_IMG=False DEBUG_PRINT=False python detect_web.py


