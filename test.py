import torchvision
import cv2
import numpy as np
import json

def PIL2cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        self.id2name = self.get_name()

    def get_name(self):
        data = {}
        for cat in self.data["categories"]:
            data[cat['id']] = cat['name']
        return data

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        img = PIL2cv(img)
        for tar in target:
            print(tar)
            box = tar['bbox']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]),
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, self.id2name[tar['category_id']], (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)




if __name__ == '__main__':
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    dt = CocoDetection('/home/shilei/Desktop/DetectionDataset/coco',
                       'zs_lvis_train.json')
    print(len(dt))
    for i in range(1000):
        dt[i]