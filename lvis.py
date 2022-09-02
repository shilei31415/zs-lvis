import json
from tqdm import tqdm
# coco datasets https://zhuanlan.zhihu.com/p/461488682

class zsLVIS:
    def __init__(self, img_path, train_json, val_json):
        self.img_path = img_path
        self.train_json = train_json
        self.val_json = val_json

        print("loading json file...")
        with open(self.train_json, 'r') as f:
            self.train_data = json.load(f)
        with open(self.val_json, 'r') as f:
            self.val_data = json.load(f)

    def build_licenses(self):
        """
        coppy info and licenses from old dataset to new
        """
        zs_train, zs_val = {}, {}
        zs_train["info"] = self.train_data["info"]
        zs_val["info"] = self.val_data["info"]

        zs_train["licenses"] = self.train_data["licenses"]
        zs_val["licenses"] = self.val_data["licenses"]

        zs_val["annotations"] = []
        zs_val["images"] = []

        zs_train["annotations"] = []
        zs_train["images"] = []
        return zs_train, zs_val

    def divide_by_frequence(self, train_f, val_f, repeat:bool=False):
        """
        devide the LVIS dataset into
        886(frequent+commom) and 377(rare)
        train_f: ['f', 'c']
        val: ['r']
        """
        zs_train, zs_val = self.build_licenses()
        train_cat, val_cat = self.divide_cat(train_f, val_f)
        zs_train["categories"], zs_val["categories"] = train_cat, val_cat

        train_id = [cat['id'] for cat in zs_train["categories"]]
        val_id = [cat['id'] for cat in zs_val["categories"]]

        print("dividing train_data ....")
        val_img_id, val_annos = self.divide_anno(self.train_data, val_id)
        val_img = [img for img in self.train_data["images"] if img['id'] in val_img_id]
        zs_val['images'] += val_img
        zs_val["annotations"] += val_annos
        print("zs_val_anno: ", len(zs_val['annotations']))
        print("zs_val_images: ", len(zs_val['images']))

        if repeat:
            train_img_id, train_annos = self.divide_anno(self.train_data, train_id, val_img_id)
        else:
            train_img_id, train_annos = self.divide_anno(self.train_data, train_id)
        train_img = [img for img in tqdm(self.train_data["images"], desc="moving img") if img['id'] in train_img_id]
        zs_train['images'] += train_img
        zs_train["annotations"] += train_annos
        print("zs_train_anno: ", len(zs_train['annotations']))
        print("zs_train_images: ", len(zs_train['images']))

        print("dividing val_data ....")
        val_img_id, val_annos = self.divide_anno(self.val_data, val_id)
        val_img = [img for img in self.val_data["images"] if img['id'] in val_img_id]
        zs_val['images'] += val_img
        zs_val["annotations"] += val_annos
        print(len(zs_val['annotations']))
        print("zs_val_anno: ", len(zs_val['annotations']))
        print("zs_val_images: ", len(zs_val['images']))

        if repeat:
            train_img_id, train_annos = self.divide_anno(self.val_data, train_id, val_img_id)
        else:
            train_img_id, train_annos = self.divide_anno(self.val_data, train_id)
        train_img = [img for img in tqdm(self.val_data["images"], desc="moving img") if img['id'] in train_img_id]
        zs_train['images'] += train_img
        zs_train["annotations"] += train_annos
        print("zs_train_anno: ", len(zs_train['annotations']))
        print("zs_train_images: ", len(zs_train['images']))

        self.save(zs_val, "zs_lvis_val.json")
        self.save(zs_train, "zs_lvis_train.json")


    def divide_cat(self, train_f, val_f):
        '''
        divide categories by frequent
        :param train_f: train_frequent
        :param val_f: val_frequent
        '''
        train_cat = []
        val_cat = []
        cats = self.train_data["categories"]
        for cat in tqdm(cats, desc="dividing categories"):
            if cat['frequency'] in train_f:
                train_cat.append(cat)
            if cat['frequency'] in val_f:
                val_cat.append(cat)

        print("train class: ", len(train_cat))
        print("val class: ", len(val_cat))
        return train_cat, val_cat

    def divide_anno(self, data, class_id, repeat=None):
        '''
        divide annotation by class_id
        :param data: old dataset
        :param class_id: class_id for new dataset
        '''
        img_id, annos = [], []

        for ann in tqdm(data["annotations"]):
            if ann['category_id'] in class_id:
                if repeat is None:
                    img_id.append(ann["image_id"])
                    annos.append(ann)
                else:
                    if img_id in repeat:
                        continue
                    img_id.append(ann["image_id"])
                    annos.append(ann)

        return img_id, annos

    def save(self, data, name):
        with open(name, "w") as f:
            f.write(json.dumps(data))




if __name__ == '__main__':
    zsdt = zsLVIS('/home/shilei/Desktop/DetectionDataset/coco',
                  '/home/shilei/Desktop/DetectionDataset/coco/annotations/lvis_v1_train.json',
                  '/home/shilei/Desktop/DetectionDataset/coco/annotations/lvis_v1_val.json')

    zsdt.divide_by_frequence(['f', 'c'], ['r'], False)