import numpy as np
import cv2
import torch
# import glob
# import os
import time
# import csv
import http
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
    'total': ['sessile serrated lesion', 'adenoma', 'hyperplastic polyp'],
    'cloud': ['cloud', 'no_cloud'],
    'dark': ['dark', 'no_dark'],
    'dark_v': ['dark_v', 'no_dark_v'],
    'irregular': ['irregular', 'no_irregular'],
    'tubular': ['tubular', 'no_tubular'],
    'vague': ['vague', 'no_vague'],
    # 'spots': ['spots', 'no_spots'],
}
COMMENTS = {  # [type_name]: [neg_list, pos_list], both lists are [0-0.33, 0.33-0.66, 0.66-1]
    'total': [
        '判断为锯齿状病变',
        '判断为腺瘤',
        '判断为增生性息肉',
    ],
    'dark': [[
        '病变颜色与背景粘膜差异不大，可能因为光线与拍摄角度难以分辨',
        '病变颜色和背景粘膜区别很小，可以认为一样',
        '病变颜色和背景粘膜一致，难以区分',
    ], [
        '病变颜色与背景粘膜差异不大，可能因为光线与拍摄角度难以分辨',
        '病变颜色和背景粘膜存在差异，较背景粘膜更深',
        '病变颜色明显比背景粘膜深',
    ]],
    'irregular': [[
        '病变形状尚且规则，可能因拍摄角度存在误差',
        '病变形状比较规则，对称',
        '病变形状十分规则，对称',
    ], [
        '病变形状轻微不规则，可能因拍摄角度存在误差',
        '病变形状不规则，可以看到分叶或是不同方向生长情况不一',
        '病变形状明显不规则，表面分叶结构清晰可见',
    ]],
    'vague': [[
        '尚能从背景粘膜中划分出病变区域',
        '病变与背景粘膜间可以划分出分界线',
        '病变与背景粘膜间可见明显、清晰的分界线',
    ], [
        '病变可能和背景粘膜融合，不易区分',
        '很难从背景粘膜中精确划出病变区域，在某些角度，病变与背景粘膜的分界难以划分',
        '病变与背景粘膜互相融合，完全无法指出分界线',
    ]],
    'dark_v': [[
        '病变中可能没有明显的异常深色血管',
        '在病变的大部分区域没有观察到异常的深色血管存在',
        '病变中完全没有异常的深色血管',
    ], [
        '病变中可能存在深色血管',
        '病变中可以看到围绕腺管的异常深色血管',
        '病变中可以看到十分明显的异常深色血管',
    ]],
    'cloud': [[
        '病变中可能没有云雾状表面结构，可能受光线与观察角度影响',
        '在病变的大部分区域都没有观察到云雾状边缘',
        '病变完全没有云雾状边缘',
    ], [
        '病变中可能存在云雾状表面结构，可能受光线与观察角度影响',
        '在病变中可以看到云雾状表面结构',
        '在病变中可以看到清晰、明显的云雾状表面结构',
    ]],
    'spots': [[
        '病变中可能没有隐窝内黑点，可能受光线与观察角度影响',
        '在病变的大部分区域都没有观察到隐窝内异常黑点',
        '完全不可见隐窝内黑点',
    ], [
        '病变中可能存在隐窝内黑点，可能受光线与观察角度影响',
        '在病变中可以看到隐窝内黑点',
        '在病变中可以看到清晰、明显的隐窝内黑点',
    ]],
    'tubular': [[
        '病变中可能没有管状/树枝状的异常腺管，可能受光线影响',
        '在病变的大部分区域都没有观察到管状/树枝状异常腺管',
        '病变中完全不存在管状/树枝状异常腺管',
    ], [
        '病变中可能存在管状/树枝状的异常腺管，可能受光线影响',
        '病变中存在管状/树枝状异常血管，被深色血管包围',
        '病变中可见明显管状/树枝状异常血管，被深色血管包围，清晰可见',
    ]],
}

def inference_all(image_url):
    result = {}
    
    print('Loading image...')
    for trial in range(3):
        try:
            response = request.urlopen(image_url, timeout=3).read()
            break
        except http.client.IncompleteRead:
            continue

    # image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imdecode(np.asarray(bytearray(response), dtype="uint8"), cv2.IMREAD_COLOR)
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

            if model_type == 'total':
                result['comment'] = COMMENTS[model_type][outputs[0]['labels'][max_score_idx]-1]
            else:
                if max_score < 1/3:
                    if pred_class == 'no_' + model_type:
                        result['comment'] = COMMENTS[model_type][0][0]
                    else:
                        result['comment'] = COMMENTS[model_type][1][0]
                elif max_score < 2/3:
                    if pred_class == 'no_' + model_type:
                        result['comment'] = COMMENTS[model_type][0][1]
                    else:
                        result['comment'] = COMMENTS[model_type][1][1]
                else:
                    if pred_class == 'no_' + model_type:
                        result['comment'] = COMMENTS[model_type][0][2]
                    else:
                        result['comment'] = COMMENTS[model_type][1][2]
            
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