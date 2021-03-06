# cuFE
This is the code accompanying the paper "cuFE: High Performance Implementation of Inner-Product Functional Encryption on GPU Platform with Application to Support Vector Machine". 

# Introduction
Privacy preservation is an emerging requirement in many applications, which becomes increasingly important in this highly connected era. Functional encryption allows the user to retrieve the results of computing a function without revealing the plaintext to the third party, effectively protecting the user's privacy. However, functional encryption is still time-consuming in the practical deployment, especially when it is applied to machine learning applications that involve huge amount of data. In this paper, a high performance GPU implementation of inner-product functional encryption (IPFE) is presented. Novel techniques are proposed to parallelize the Gaussian sampling, which one of the most time-consuming operations in the IPFE. A systematic investigation was also carried out to select the best strategy for implementing NTT and INTT for different security levels. This repository also contain source codes for implementing Support Vector Machine with Inner Product Functional Encryption, using the proposed gaussian sampling and Number Theoretic Transform on GPU.

# How to use
There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, which is suitable for RTX2060,RTX2070 and RTX2080.

0) This source code provides the prediction of SVM.

    We need to get a model file after training a dataset by libsvm library. 
    
    The dataset is in https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    
    We can get a training program called svm-train by compiling the source code from the libsvm library. The command for training is as follows:

    $ ./svm-train (data file)

1) The main function calls two kinds of prediction functions.

    predict() is used for three purposes: SVM classification without encryption, SVM +IPFE executed on a CPU, and the naive version IPFE on a GPU.

    predict2() is used for the optimized and merged version of IPFE on a GPU.

    * You may also comment out one of the prediction functions for testing.

2) Each prediction function calls one of the svm_predict functions.

    Original SVM: use svm_predict(). It is the same with the libsvm library.

    No encryption: use svm_predict2(). It is used with encoding to compare with the IPFE version.

    IPFE on CPU: use svm_predict3(). You may enable AVX2 support in the param.h file.
    
    "#define AVX2" (line 7)

    The naive version IPFE on GPU: use svm_predict4(). 

    The parallel version of IPFE on GPU: use svm_predict5(). There are two kinds of key generation and decryption functions. "_gui2" is the merged version and "_gui3" is the optimized (not merged) version.
    
    * You may also comment out one of the svm_prediction functions for testing.

3) Run the following commands to test run the SVM classification protected by cuFE.

    $ ulimit -s unlimited

    $ ./svm_ipfe-gpu (data file) (model file) (output file)
