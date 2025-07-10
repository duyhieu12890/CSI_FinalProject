#!/bin/bash

cloudflared tunnel --url http://127.0.0.1:8000 2>&1 | tee tunnel.log
