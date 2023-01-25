#!/usr/bin/bash

cd /g/g20/eljurf1/weave_demos/ball_bounce/03_simulation_ensembles/themis_ensembles/setup/.ens_5_setup/execution_1
/g/g20/eljurf1/weave_demos/ball_bounce/ball_bounce_demo_venv/bin/python /g/g20/eljurf1/weave_demos/ball_bounce/ball_bounce_demo_venv/lib/python3.7/site-packages/themis/backend -t0 -a1 -p0 -e0 --setup-dir /g/g20/eljurf1/weave_demos/ball_bounce/03_simulation_ensembles/themis_ensembles/setup/.ens_5_setup -c 100 >> /g/g20/eljurf1/weave_demos/ball_bounce/03_simulation_ensembles/themis_ensembles/setup/.ens_5_setup/execution_1/themis_backend.log 2>&1
wait
