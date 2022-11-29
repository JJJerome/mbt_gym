#!/bin/bash
/etc/init.d/postgresql start
jupyter notebook --no-browser --port $1 --ip=0.0.0.0 --allow-root