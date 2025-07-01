import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import base64
import re
import os
from io import BytesIO
from PIL import Image
import pdb
import tempfile
import ast

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

class ZonUIModel():
    def load_model(self, model_name_or_path="zonghanHZH/ZonUI-3B"):
        self.min_pixels = 256*28*28
        self.max_pixels = 1280*28*28
        self.device = "cuda"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    
    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=99999999,
        )
        # print("Original image size: {}x{}".format(image.width, image.height))
        # print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element."},
                    {"type": "image", "image": image_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    {"type": "text", "text": instruction}
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        
        inputs = self.processor(
            text=[text],
            images=[resized_image],
            return_tensors="pt",
            training=False
        ).to(self.device)

        generated_ids = self.model.generate(**inputs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        # print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            coordinates = ast.literal_eval(response)
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()
