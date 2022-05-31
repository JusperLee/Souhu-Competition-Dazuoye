# 2022 Sohu Campus Emotion Analysis × Recommendation Sorting Algorithm Competition

```
├── README
├── Task1
│   ├── config.py
│   ├── datasets.py
│   ├── dice_loss.py
│   ├── instance.py
│   ├── model.py
│   ├── preprocess.py
│   ├── process_data.py
│   ├── souhu_AE.py
│   ├── souhu_baseline.py
│   ├── souhu_diceloss.py
│   ├── souhu_fgm.py
│   ├── souhu_gradnorm.py
│   ├── souhu_multicls_imp_longformer.py
│   ├── souhu_multicls_longformer.py
│   ├── souhu_multicls_longformer_fgm.py
│   ├── souhu_multicls_multimodel.py
│   ├── submit.sh
│   ├── test.py
│   ├── train.py
│   ├── trainer.py
│   └── utils.py
├── Task2
│   ├── README.md
│   ├── baseline-session.py
│   ├── baseline-torch-tf.py
│   ├── baseline-torch.py
│   ├── baseline.py
│   ├── cat_item_embs.py
│   ├── comm.py
│   ├── config.py
│   ├── dataloader.py
│   ├── dataloader_new.py
│   ├── evaluation.py
│   ├── features.py
│   ├── hash.py
│   ├── item2vec_main.py
│   ├── item2vec_model.py
│   ├── model.py
│   ├── params.py
│   ├── save_hash.py
│   ├── sub.sh
│   ├── task02_recommendation.ipynb
│   ├── test.ipynb
│   ├── test_dataloader_new.py
│   ├── train.py
│   └── utils.py
```

## How to train

### In Task1

Best performance train and testing

```shell
python souhu_multicls_imp_longformer.py
```

### In Task2

Best performance train and testing

```shell
python baseline-torch-tf.py --batch_size=2048 --info=25epoch-20decay-bs2048
```

## Results

### Task1

![Task1 Results](./img/task1_results.png)

### Task2

![Task1 Results](./img/task2_results.png)

### ALL Results

![Task1 Results](./img/all.png)

## Citing code

```bibtex
@misc{LRS3SS,
  author = {Kai Li, Chaoqun He, Kuiyu Wang, Shenghao Yang},
  title = {Souhu-Competition-Dazuoye},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JusperLee/Souhu-Competition-Dazuoye}},
}
```

## License
MIT License

Copyright (c) 2022 Kai Li (李凯)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
