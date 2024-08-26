#!/usr/bin/env bash

# Prepares the store for the study


# Copy store with experimental data

cp experiments/pyranda.sql .


# For AWS-based studies
if [ -d /shared_data ]; then
   cp experiments/pyranda.sql /shared_data
fi

