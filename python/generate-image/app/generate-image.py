from datetime import datetime
from PIL import Image

import time
import os
import uuid
import requests

def generate_ai_image(prompt, save_path):
    # make unique prompt
    prompt += str(uuid.uuid4())  
    formatted_prompt = prompt.replace(" ", "-")

    # url to create ai image: https://image.pollinations.ai/prompt/...
    url = f"https://image.pollinations.ai/prompt/{formatted_prompt}"

    while True:
        try:
            response = requests.get(url)

            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                image_save_name = f"img_{timestamp}.png"
                image_save_path = os.path.join(save_path, image_save_name)

                os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                
                with open(image_save_path, 'wb') as f:
                    f.write(response.content)
                
                # crop watermark/artifacts
                image = Image.open(image_save_path)
                image = image.crop((0, 0, image.width, image.height - 48))

                cropped_image = image.crop((0, 0, image.width, image.height - 100))
                cropped_image.save(image_save_path)

                # show generated image using default 
                # image visor
                print(f"Showing image {image_save_path} ...")
                image.show()

                return image_save_path
            else:
                print(f"Retrying... Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed: {e}. Retrying...")

        time.sleep(5)  

if __name__ == "__main__":
    print("")
    print("")
    
    user_prompt = input("Enter a prompt to create an image: ")
    prompts = [ user_prompt ]
    
    save_path = "ai-images"
    
    # create folder if it does not exist yet
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    for prompt in prompts:
        image_path = generate_ai_image(prompt, save_path)

    print("")    

