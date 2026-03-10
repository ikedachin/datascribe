from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success
def get_prompts(prompts_settings):
    prompts_dict = {}
    for prompt_path_dict in prompts_settings:
        key, prompt_path = list(prompt_path_dict.items())[0]
        print(msg_debug(f"Loading prompt for {key} from {prompt_path}"))
        with open(prompt_path, "r", encoding='utf-8') as f:
            ocr_prompt = f.read()
        prompts_dict[key] = ocr_prompt
    # print(msg_success(f"Loaded Prompts in commons: {prompts_dict}"))
    return prompts_dict