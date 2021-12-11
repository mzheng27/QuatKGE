import config
from  models import *
import json
import os 
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_work_threads(8)
con.set_checkpoint_dir("./checkpoint/FB15K/kernel/")
con.set_result_dir("./result/")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_test_model(QuatE, path="./checkpoint/FB15K/kernel/QuatE-49.ckpt")
con.link_prediction()
con.triple_classification()

