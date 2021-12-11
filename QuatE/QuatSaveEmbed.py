import config
from  models import *
import json
import os 
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")  #dataset
con.set_result_dir('/home/exx/Documents/QuatE-master/result/')
con.init()
con.set_test_model(QuatE, path="/home/exx/Documents/QuatE-master/checkpoint/dot_product_originalhp/QuatE-9999.ckpt")
con.save_embedding_matrix(con.testModel.state_dict())

