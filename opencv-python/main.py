#!/usr/bin/env python3

import numpy as np
from control.Planner import Planner
import json

stream = open('params.json')
params = json.load(stream)

planner = Planner(params)

planner.start()
