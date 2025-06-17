import json


filename = '/llm_reco/dehua/data/transfer_data/food_finetune_data/food2k_finetune_raw.json'

with open(filename, 'r') as file:
    data = json.load(file)

new_d_list = []
for d in data:
    images = d["images"][0].replace("/map-vepfs/dehua/data/data/", "/llm_reco/dehua/data/food_data/")
    new_d = d
    new_d["images"] = [new_d]
    new_d_list.append(new_d)


new_filename = "/llm_reco/dehua/data/transfer_data/food_finetune_data/food2k_finetune_raw.json"
with open(new_filename, 'w') as file1:
    json.dump(new_d_list, file1)
