import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def main():
    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,
        cache_dir="./cache/huggingface/hub",
    )
    vl_gpt = vl_gpt.eval()

    image = "./images/ve.png"
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\nWhat does the dog on the right has to do to look similar to the one in the left? Ignore the captions and answer only based on the photos. What injectable substances might make the transformation possible?",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    while True:

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)

        question = input("Please type your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        conversation.append({"role": "<|User|>", "content": question, "images": []})
        conversation.append({"role": "<|Assistant|>", "content": ""})

if __name__ == '__main__':
    main()