#!/usr/bin/env python

import numpy as np
from control.Planner import Planner
import json

stream = open('params.json')
params = json.load(stream)

planner = Planner(params)

planner.start()
