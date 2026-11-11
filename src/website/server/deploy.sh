#!/bin/bash

gcloud builds submit --tag gcr.io/kilterprediction/flask-fire
gcloud run deploy flask-fire \
        --image gcr.io/kilterprediction/flask-fire \
        --region us-central1 \
        --memory 2Gi \
        --cpu 2 \
        --cpu-boost \
        --timeout 300
firebase deploy