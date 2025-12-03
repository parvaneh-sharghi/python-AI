import os 
import argparse 
import pickle 
from typing import List

import numpy as np


from dotenv import load_dotenv
import os

load_dotenv()

api = os.getenv("OPENAI_API_KEY")
print(api)