import cv2
import numpy as np
import onnxruntime as ort


norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def softmax(x: np.ndarray, axis: int=-1):
    """
    Equivalent to torch.nn.functional.softmax()
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class APGCCModel:
    def __init__(self, model_path):
        
        self.model = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider']
        )
        
        self.inp_name = [x.name for x in self.model.get_inputs()]
        self.opt_name = [x.name for x in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and prepare the image for ONNX inference.

        Args:
            image (np.ndarray): Input image in shape (H, W, C) - OpenCV format

        Returns:
            input_tensor (np.ndarray): (1, C, H, W), float32
            ratio (tuple): (w_ratio, h_ratio) for resizing back (postprocess)
        """
        h_model, w_model = self.model_inpsize[1], self.model_inpsize[0] 
        h_original, w_original = image.shape[:2]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.model_inpsize)
        image = image.astype(np.float32) / 255.
        image = (image - norm_mean) / norm_std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        
        ratio_w = w_original / w_model
        ratio_h = h_original / h_model

        return image, (ratio_w, ratio_h)
        
    def postprocess(self, predict, ratio, threshold=0.5):
        output_score = softmax(predict[0], axis=-1)[:, :, 1][0]
        onnx_points = predict[1][0]
        filtered_points = onnx_points[output_score > threshold]
        pcount = int(np.sum(output_score > threshold))
        
        filtered_points[:, 0] *= ratio[0]  # x
        filtered_points[:, 1] *= ratio[1]  # y
        
        return filtered_points.tolist(), pcount
            
    
    def run(self, image: np.ndarray, threshold: int=0.5):
        inp, ratio = self.preprocess(image)
        
        predict = self.model.run(self.opt_name, {self.inp_name[0]: inp})
        point, predict_cnt = self.postprocess(predict, ratio, threshold)

        return [point, predict_cnt]
    
if __name__ == '__main__':
    import os
    import glob
    
    os.makedirs("output/predicts", exist_ok=True)
    
    model = APGCCModel('weights/SHHA_best.onnx')
    for sample in glob.glob('../5_img/test/*.jpg'):
        name = sample.split('/')[-1]
        img = cv2.imread(sample)
        points, predict_cnt = model.run(img, threshold=0.5)
        print("Image path: ", sample, "\tcount: ", predict_cnt)
        if predict_cnt > 0:
            for p in points:
                x, y = int(p[0]), int(p[1])
                img = cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
                
            save_path = os.path.join("output/predicts", f"{name}")
            cv2.imwrite(save_path, img)