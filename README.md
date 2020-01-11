# TANN model

This is the code for paper [Exploiting Topic-based Adversarial Neural Network for Cross-domain Keyphrase Extraction](https://ieeexplore.ieee.org/document/8594884) (ICDM 2018)

If you use the code, please kindly cite the paper:
```
@inproceedings{wang2018exploiting,
  title={Exploiting Topic-based Adversarial Neural Network for Cross-domain Keyphrase Extraction},
  author={Wang, Yanan and Liu, Qi and Qin, Chuan and Xu, Tong and Wang, Yijun and Chen, Enhong and Xiong, Hui},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={597--606},
  year={2018},
  organization={IEEE}
}
```
## Usage
* data

  sampled data is provided in sample_data directory, you can prepare your own data including source domain and target domain data
  
  vocab.pickle can be obtained using voc.py.
  
* you can train the model using the following script, more parameters can be turned in adv_train.py
    ``` javascript
    CUDA_VISIBLE_DEVICES=0 python adv_train.py \
    --log_dir 'your log path' \
    --CORPUS 'source_domain,target_domain' \
    --batch_size 64 \
    --dropout_keep_prob 0.5 \
    --lstm_dim 300 \
    --num_epochs 100 \
    --num_filters 200 \
    --word_dim 300 \
    --evaluate_every 50 \
    --conv_activation "elu" \
    --early_stop 15 \
    --use_gate 1 \
    --lr 0.1 \
    --clip 1 \
    --lamda_type 3 \
    --num_decode_steps 100 \
    --lm_rate 0.2 \
    --topic_num 50
    ```
