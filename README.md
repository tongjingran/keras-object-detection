# keras-object-detection
Various models for Object Detection with tensorflow backend keras.

### Tested ENV
OS: Ubuntu 16.04

GPU: GTX 1080 TI

GPU_driver: nvidia - 384.111

##### modules
| module_name | version |
|:-------------:|:---------:|
| tensorflow-gpu | 1.7.0 |
| scikit-learn | 0.19.1 |
| opencv (built) | 3.4.0 |
| numpy | 1.14.2 |

### Dataset
You can download Pascal VOC dataset by following commands.
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar  
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

train data : voc2012_train + voc2012_val + voc2007_train

val data   : voc2007_val

test data  : voc2007_test

### results

