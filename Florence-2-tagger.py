from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy

# 下载Florence-2模型替换路径
model_id = "/mnt/d/models/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def clean_caption(caption):
    remove_phrases = ["The image shows", "In this image", "This image shows", "This is an image of"]
    for phrase in remove_phrases:
        if caption.lower().startswith(phrase.lower()):
            caption = caption[len(phrase):].strip()
            if caption.startswith(",") or caption.startswith("."):
                caption = caption[1:].strip()
    return caption

def generate_caption(image_path, task_prompt):
    try:
        # 尝试打开图像
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print(f"无法识别图像文件：{image_path}，跳过。")
        return None

    # 生成输入
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # 生成文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 解析生成结果
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    field_key = task_prompt.strip('<>').upper()
    caption = parsed_answer.get(f'<{field_key}>', "")

    # 清理 Caption
    caption = clean_caption(caption)
    return caption

# 遍历需要打标的文件夹
image_folder = "/mnt/d/datasets/style-icons/Cute Color/all"
# DETAILED_CAPTION是生成比较详细的caption，CAPTION是生成比较简略的caption
task_prompt = '<DETAILED_CAPTION>'
# task_prompt = '<CAPTION>' 

for file_name in os.listdir(image_folder):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, file_name)

        # 检查是否已有 Caption 文件
        caption_file_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_file_path):
            print(f"Caption 已存在，跳过：{caption_file_path}")
            continue

        # 生成 Caption
        caption = generate_caption(image_path, task_prompt)
        if caption:
            print(f"Caption for {file_name}: {caption}")
            # 保存 Caption 到同名 .txt 文件
            with open(caption_file_path, "w", encoding="utf-8") as f:
                f.write(caption)

print("所有图像的 Caption 生成并保存完成。")