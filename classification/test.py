
def run(rank, world_size):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(ori_processor_path) 

    model = model.to(torch.device(rank))
    model = model.eval()

    ### get categories name
    with open('./val_data/fgvc_aircraft.txt', 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    print(len(categories))
    print(categories)   ### 对应 0-101

    ### get validation data
    pth_file_path = './val_data/fgvc_aircraft.pth'
    predictions = torch.load(pth_file_path)
    
    val_set = []
    for item in predictions:
        for k,v in item.items():
            val_set.append({k:int(v['label'])})
    print(len(val_set))
    print(val_set[0])


    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(val_set)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = val_set[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    ### 遍历 val 中的所有图片
    error_count = 0
    right_count = 0
    for image in tqdm(split_images): 
        ### 获取图片信息
        for k,v in image.items():
            image_path = k
            image_label = v
        image_cate = categories[image_label]   
        # plot_images([image_path])
    
        question = (
        "This is an image containing an aircraft. Please identify the model of the aircraft based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>species name</answer>\n"
        "Please strictly follow the format."
        )
    
        image_path = image_path
        query = "<image>\n"+question
        # print(RED+query+RESET)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # print("\033[92m" + response + "\033[0m")
    
        try:
            match = re.search(r"<answer>(.*?)</answer>", response)
            answer_content = match.group(1)
            # print(image_cate, answer_content)
            image_cate = image_cate.replace(' ','').replace('_','').lower()
            answer_content = answer_content.replace(' ','').replace('_','').lower()
            # judgement
            if image_cate in answer_content or answer_content in image_cate:
                print('yes')
                right_count += 1
                logger.info('Local Right Number: ' + str(right_count))
            else:
                print('no')
        except Exception as e:
            error_count+=1
            
    return [error_count, right_count]
