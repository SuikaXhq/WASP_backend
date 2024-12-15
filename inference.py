import numpy as np
import cv2
import torch
# import glob
# import os
import time
# import csv
import urllib.request as request

from model import create_model  # Ensure this is correctly imported from your model file
# from config import NUM_CLASSES, DEVICE, CLASSES, MODEL_PATH, TEST_DIR, INFERENCE_OUTPUT_CSV # Ensure these are correctly defined in your config file
# from prediction_reviewer import review

# input: image url, model type
# output: bounding box, score, class
# step 1 - load image with imdecode
# step 2 - preprocess image
# step 3 - load model
# step 4 - predict
# step 5 - get bounding box, score, class

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_TYPES = {
    'total': ['adenoma', 'sessile serrated lesion', 'hyperplastic polyp'],
    'cloud': ['cloud', 'no_cloud'],
    'dark': ['dark', 'no_dark'],
    'dark_v': ['dark_v', 'no_dark_v'],
    'irregular': ['irregular', 'no_irregular'],
    'tubular': ['tubular', 'no_tubular'],
    'vague': ['vague', 'no_vague'],
    # 'spots': ['spots', 'no_spots'],
}

def inference_all(image_url):
    result = {}
    
    print('Loading image...')
    response = request.urlopen(image_url)
    # image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imdecode(np.asarray(bytearray(response.read()), dtype="uint8"), cv2.IMREAD_COLOR)
    # save image
    # cv2.imwrite(f'inference_outputs/0.jpg', image)
    # orig_image = image.copy()
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    print('Image loaded and preprocessed.')
    
    for model_type in MODEL_TYPES:
        result[model_type] = inference(image_url, model_type, image=image)
    return result

def inference(image_url, model_type, image=None):
    # Create a different color for each class
    # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load the best model and trained weights
    
    CLASSES = ['__background__'] + MODEL_TYPES[model_type]
    NUM_CLASSES = len(CLASSES)
    MODEL_PATH = f'model/outputs/{model_type}/best_model.pth'
    
    print('Creating model...')
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    print('Model created.')

    # Directory where all the images are present
    # DIR_TEST = TEST_DIR
    # test_images_png = glob.glob(f"{DIR_TEST}/*.png")
    # test_images_jpg = glob.glob(f"{DIR_TEST}/*.jpg")
    # test_images = test_images_png + test_images_jpg  # Combine the lists
    # print(f"Test instances: {len(test_images)}")

    # Define the detection threshold
    detection_threshold = 0.0

    # frame_count = 0
    # total_fps = 0

    # Prepare CSV file for saving bounding box, classification data, and score
    # with open(INFERENCE_OUTPUT_CSV, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["image_id", "Bounding Box", "Image Size", "classification", "Score"])

        # for i in range(len(test_images)):
    if image is None:
        print('Loading image...')
        response = request.urlopen(image_url)
        # image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imdecode(np.asarray(bytearray(response.read()), dtype="uint8"), cv2.IMREAD_COLOR)
        # save image
        # cv2.imwrite(f'inference_outputs/0.jpg', image)
        # orig_image = image.copy()
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)
        image = image.to(DEVICE)
        print('Image loaded and preprocessed.')
    print('Start inference...')
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
    end_time = time.time()
    infer_time = end_time - start_time
    print('Inference done. Time:', infer_time)
    
    

    # fps = 1 / (end_time - start_time)
    # total_fps += fps
    # frame_count += 1

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    result = {}
    result['box'] = None
    result['score'] = None
    result['class'] = None

    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        
        # Find the prediction with the highest score
        max_score_idx = scores.argmax()
        max_score = scores[max_score_idx]
        
        if max_score >= detection_threshold:
            box = boxes[max_score_idx].astype(np.int32)
            pred_class = CLASSES[outputs[0]['labels'][max_score_idx]]
            # color = COLORS[CLASSES.index(pred_class)]
            # text = f"{pred_class}: {max_score:.2f}"

            # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            # cv2.putText(orig_image, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Write the data to the CSV file, including the score
            # writer.writerow([image_name, str(box), orig_image.shape[:2], pred_class, f"{max_score:.2f}"])
            
            # result['box'] = box
            result['score'] = str(round(max_score, 6))
            result['class'] = pred_class
            
            if max_score < 1/3:
                if pred_class == 'no_' + model_type:
                    result['comment'] = '可能没有'
                else:
                    result['comment'] = '可能有'
            elif max_score < 2/3:
                if pred_class == 'no_' + model_type:
                    result['comment'] = '几乎没有'
                else:
                    result['comment'] = '可以看到'
            else:
                if pred_class == 'no_' + model_type:
                    result['comment'] = '没有迹象'
                else:
                    result['comment'] = '可见显著'
            
            # cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    result['infer_time'] = infer_time
    
    print('Inference result:', result)
    
    return result
    

    # print(f"Image {i+1} done...")
    # print('-'*50)

    # print('TEST PREDICTIONS COMPLETE')
    # cv2.destroyAllWindows()

    # avg_fps = total_fps / frame_count
    # print(f"Average FPS: {avg_fps:.3f}")

    # result = review()
    # print(result)
    

if __name__ == '__main__':
    image_url = 'https://img0.baidu.com/it/u=1165613183,262764682&fm=253&fmt=auto&app=138&f=JPEG?w=667&h=500'
    print(inference_all(image_url))
    # print(inference(image_url, 'total'))
    # print(inference(image_url, 'cloud'))
    # print(inference(image_url, 'dark'))
    # print(inference(image_url, 'dark_v'))
    # print(inference(image_url, 'irregular'))
    # print(inference(image_url, 'tubular'))
    # print(inference(image_url, 'vague'))
    # print(inference(image_url, 'spots