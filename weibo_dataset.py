import torch
import torch.utils.data as data
import pandas
from tqdm import tqdm
from transformers import BertTokenizer
import os
from PIL import Image
import cv2
from transformers import AutoFeatureExtractor,ViTImageProcessor
import data.util as util
import cn_clip.clip as clip_cn
from cn_clip.clip import load_from_name, available_models
# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_chinese_clip_model(device):
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='pretrain_models/CLIP_chinese/')
    model.eval()
    return model,preprocess

def pre_feature_extractor():
    # Load a feature extractor from the transformers library
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    return feature_extractor

# Function to read an image
def read_img(imgs, root_path, LABLEF):
    for img in imgs:
        GT_path = "{}/{}/{}".format(root_path, LABLEF, img)
        if os.path.exists(GT_path):
            try:
                img_swin = util.read_img(GT_path)
            except:
                continue
            img_GT = Image.open(GT_path).convert('RGB')
            # img_cv = cv2.imread(GT_path)
            return img_GT,img_swin,GT_path
        else:
            continue
    return False

def load_chinese_clip_process():
    model, preprocess = load_from_name("ViT-H-14", device='cpu', download_root='pretrain_models/CLIP_chinese/')
    model.eval()
    return preprocess

class weibo_dataset(data.Dataset):
    def __init__(self, root_path='./data/datasets/weibo', image_size=224, is_train=True):
        super(weibo_dataset, self).__init__()
        # clipmodel, preprocess = pre_clip()
        self.is_train = is_train
        self.root_path = root_path
        self.index = 0
        self.label_dict = []
        # self.preprocess = preprocess
        self.image_size = image_size
        self.local_path = 'data/datasets/weibo'
        self.swin = pre_feature_extractor()
        print('dataset中加载clip')
        self.clip_cn_preprocess = load_chinese_clip_process()

        # Read data from CSV file
        # wb = torch.load(self.local_path + '/{}_weibo.pt'.format('train' if is_train else 'test'))
        wb = torch.load('{}_weibo.pt'.format('train' if is_train else 'test'))
        self.label_dict = wb

    def __getitem__(self, index):
        record = self.label_dict[index]
        label, image_paths, content,caption,OCR,if_OCR,content_entities_k,image_entities_k,new_entities_k= (
            record['label'], record['image_paths'], record['content'],record['caption'], record['OCR'],record['if_OCR'],
            record['content_entities_k'],record['image_entities_k'],record['new_entities_k'])
        # Determine the label folder
        if label == 1:
            LABLE_F = 'rumor_images'
        else:
            LABLE_F = 'nonrumor_images'
        # print(images)
        imgs = image_paths.split('|')
        try:
            img_GT,image_swin,img_path = read_img(imgs, self.root_path, LABLE_F)
        except Exception:
            raise IOError("Load {} Error".format(imgs))

        image_clip_cn = self.clip_cn_preprocess(Image.open(img_path)).unsqueeze(0)

        if if_OCR == 'True':
            txt_bert = content + OCR
        else:
            txt_bert = content
        return (self.swin(image_swin, return_tensors="pt",do_rescale=False).pixel_values.cpu(),
                image_clip_cn.cpu(),content,caption,txt_bert,content_entities_k,image_entities_k,new_entities_k),label

    def __len__(self):
        return len(self.label_dict)

# Custom collate function
def collate_fn(data):
    caption = [i[0][3] for i in data]
    images_swin = [i[0][0] for i in data]
    image_clip_preprocess = [i[0][1] for i in data]
    text = [i[0][2] for i in data]
    text_bert = [i[0][4] for i in data]
    content_entities_k = [i[0][5] for i in data]
    image_entities_k = [i[0][6] for i in data]
    new_entities_k = [i[0][7] for i in data]
    labels = [i[1] for i in data]

    images_swin = torch.squeeze(torch.stack(images_swin))
    image_clip = torch.stack(image_clip_preprocess)
    labels = torch.LongTensor(labels)
    # print(len(new_entities_k))


#     text_temp = []
#     image_temp = []
#     for content_entities,image_entities in zip(content_entities_k,image_entities_k):
#         # print(content_entities.shape)
#         # print(image_entities.shape)
#         text_pad_num = 20 - content_entities.shape[0]
#         image_pad_num = 5 - image_entities.shape[0]
#         # 统一补充到20个，方便batch运算
#         text_entities_fea = torch.nn.functional.pad(content_entities, (0, 0, 0, text_pad_num), mode='constant')
#         image_entities_fea = torch.nn.functional.pad(image_entities, (0, 0, 0, image_pad_num), mode='constant')
#         text_temp.append(text_entities_fea)
#         image_temp.append(image_entities_fea)
#     new_temp = []
#     for new_entities in new_entities_k:
#         pad_num = 25 - new_entities.shape[0]
#         # 统一补充到25个，方便batch运算
#         new_entities_fea = torch.nn.functional.pad(new_entities, (0, 0, 0, pad_num), mode='constant')
#         new_temp.append(new_entities_fea)
        
#     content_entities_k = torch.stack(text_temp)
#     image_entities_k = torch.stack(image_temp)
#     new_entities_k = torch.stack(new_temp)
    return images_swin, image_clip,text,caption,text_bert,content_entities_k,image_entities_k,new_entities_k,labels