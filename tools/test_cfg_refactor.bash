#!/usr/bin/env bash

mppnet=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

mppnet_4frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet_4frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

agg_mppnet_4frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet.yaml',
    '--modify',
    'cfgs/waymo_models/modifiers/mppnet/4frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

mppnet_16frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet_16frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

agg_mppnet_16frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet.yaml',
    '--modify',
    'cfgs/waymo_models/modifiers/mppnet/16frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

mppnet_e2e_membank_inf=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet_e2e_memorybank_inference.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

agg_mppnet_e2e_membank_inf=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet.yaml',
    '--modify',
    'cfgs/waymo_models/modifiers/mppnet/e2e_memorybank_inference.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

mppnet_e2e_membank_inf_16frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet_e2e_memorybank_inference_16frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

agg_mppnet_e2e_membank_inf_16frames=$(python -c "
from train import parse_config
import sys
import yaml
sys.argv += [
    '--cfg_file',
    'cfgs/waymo_models/mppnet.yaml',
    '--modify',
    'cfgs/waymo_models/modifiers/mppnet/e2e_memorybank_inference.yaml',
    'cfgs/waymo_models/modifiers/mppnet/16frames.yaml',
]
print(yaml.dump(parse_config()[1], default_flow_style=None))
")

diff <(echo "$mppnet") <(echo "$mppnet_4frames")
diff <(echo "$mppnet_4frames") <(echo "$agg_mppnet_4frames")
diff <(echo "$mppnet_16frames") <(echo "$agg_mppnet_16frames")
diff <(echo "$mppnet_e2e_membank_inf") <(echo "$agg_mppnet_e2e_membank_inf")
diff <(echo "$mppnet_e2e_membank_inf_16frames") <(echo "$agg_mppnet_e2e_membank_inf_16frames")