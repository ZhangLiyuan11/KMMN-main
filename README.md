# KMMN
Code for paper["***Knowledge Enhanced Multimodal Multi-grained Network for Fake News Detection***"]

### Environment
please refer to the file requirements.txt.

### Data Processing
We use publicly available APIs to extract background knowledge of news. You can apply for your key [here](https://cloud.baidu.com/product/nlp_basic/entity_analysis), and we use this [tool](https://cloud.baidu.com/product/imagerecognition/general) to extract entities from images. The program for using APIs is already prepared in the project.
You can also directly use our processed embeddings, and we share this [data](https://drive.google.com/file/d/12op769C_vmli9y2Vw3dkTHFE-9QA7bJC/view?usp=sharing) through anonymous Google Drive accounts.
Pre-trained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
After placing the data, start training the model:
```python
python main.py
```

