# JigsawNet

JigsawNet is an image fragment reassembly system based on convolutional neural network. It is able to robustly reassemble irregular shredded image fragments.

This repository includes CNN-based pairwise alignment measurement and loop-closure based global composition. Please check our published paper for more algorithm details (https://arxiv.org/pdf/1809.04137.pdf).

There are some of reassembly results from various public datasets and our own datasets (1st, 2nd, 3rd and 4th row contains 9, 36, 100, 376 piece fragments repectively). More results are demonstrated in our paper.

![demon](https://raw.githubusercontent.com/Lecanyu/JigsawNet/master/Examples/demo.png)





# 1. Prerequisites

We have tested this code on Windows10 x64 operation system.
We developed a CNN to measure pairwise compatibility, which has been implemented on Python 3.6 and tensorflow.
To globally reassemble, we designed a loop-based composition to calculate consistently reassembly result. The global algorithm has been implemented in C++ on Microsoft Visual Studio 2015 with CUDA 8 or 9 support. 

Below dependencies are necessary to run pairwise compatibility measurement module.
* Python 3.6
* Tensorflow 1.7.0 and its dependencies

You should install below software or libraries to run global reassembly part.
* OpenCV 3.4.1
* Eigen 3.3.4
* CUDA 8.0 or 9.0

Other version of those dependencies/libraries have not tested.

If you want to compile or run this code on different OS or environments, a few of modifications will be needed.



# 2. Run pairwise compatibility measurement

We provide three modes to drive the scripts. 

Training
------------
Train network on training dataset. Please read the code to make sure all of pathes are set correctly.
```
python boost.py -m training
```

batch_testing
------------
Test network performance on tfrecord. You should prepare the testing data before you run.
```
python boost.py -m batch_testing
```

single_testing
------------
Test network performance on image data. You just need to prepare the pairwise alignment file. The image merging will be done by our program.
```
python boost.py -m single_testing
```

We recommend you reading the code to figure out how to modify the path and tune whatever you want. We think the structure of code is relatively clear and it should be self-explanatory.

You can find data/file format demonstration in the 'Examples' folder.


# 3. Run global reassembly

Global reassembly module is developed in C++ with CUDA support. For the algorithm details, please check our paper.

You can find a running example (run.bat) in the 'Examples' folder. 
If you successfully compile this module, you can run the example data by following the corresponding data format.


# 4. Run the example data

Put your own compiled GlobalReassembly.exe into Examples folder, and then run the .bat file to reassembly.

If everything goes well, you will get several output files (filtered_alignments.txt, pose_result_x.txt, reassembled_result_x.png and selected_transformation.txt)

We have already put those output files into the folders for refering. 


# 5. Run your own datasets

To solve your own jigsaw puzzles, a pairwise alignment calculation module is needed. In our experiments, we use an [existing method](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.683.4733&rep=rep1&type=pdf) to calculate pairwise alignment candidates. But you can use other more fancy algorithms to do it.

The pairwise alignments are represented by a list of 3x3 rigid transformation matrix. Please check our example data for the format details.


# 6. Datasets and pre-trained net parameters
Our experiment datasets and pre-trained model can be downloaded [here](https://drive.google.com/open?id=1sUIcAzFTJNAAEEhqdYAKMKgzjVwRvsP4).

From this link, you can find 5 different datasets (one for training and four for testing) and the JigsawCNN parameters checkpoint which has been trained from the training dataset. 

You can directly load this checkpoint to run the example data. 

Note: For successfully load the checkpoint on your machine, you should modify the checkpoint file to correct path (i.e. JigsawCNN_checkpoint/g0/checkpoint, JigsawCNN_checkpoint/g1/checkpoint, ...). 
Since this code has been implemented on tensorflow, and the pretrained parameters can only be used on tensorflow library.




# 7. Citation
If our work is useful in your research, please cite 

```
@article{le2018jigsaw,
  title={JigsawNet: Shredded Image Reassembly using Convolutional Neural Network and Loop-based Composition},
  author={Le, Canyu and Li, Xin},
  journal={arXiv preprint arXiv:1809.04137},
  year={2018}
}
```
