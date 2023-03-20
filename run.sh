#!/bin/bash

exec python3 backend.py &
exec python3 -m streamlit run frontend.py
