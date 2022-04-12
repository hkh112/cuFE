# cuFE
This is the code accompanying the paper "cuFE: High Performance Implementation of Inner-Product Functional Encryption on GPU Platform with Application to Support Vector Machine". 

# Introduction
Privacy preservation is an emerging requirement in many applications, which becomes increasingly important in this highly connected era. Functional encryption allows the user to retrieve the results of computing a function without revealing the plaintext to the third party, effectively protecting the user's privacy. However, functional encryption is still time-consuming in the practical deployment, especially when it is applied to machine learning applications that involve huge amount of data. In this paper, a high performance GPU implementation of inner-product functional encryption (IPFE) is presented. Novel techniques are proposed to parallelize the Gaussian sampling, which one of the most time-consuming operations in the IPFE. A systematic investigation was also carried out to select the best strategy for implementing NTT and INTT for different security levels. This repository also contain source codes for implementing Support Vector Machine with Inner Product Functional Encryption, using the proposed gaussian sampling and Number Theoretic Transform on GPU.

# How to use
There is a Makefile accompanied with the source codes in each separate folder. You can build the executable by typing "make".

Note that you need to change the sm version in GPU to suit your device. The default is -arch=sm_75, which is suitable for RTX2060,RTX2070 and RTX2080.

0) This source code provides the prediction of SVM.

    We need to get a model file after training by libsvm library.

1) The main function call two kinds of prediction functions.

    predict() is for no encryption, IPFE on CPU, the naive version IPFE on GPU.

    predict2() is for the parallel version of IPFE on GPU.

    * You may also comment out one of the prediction functions for testing.

2) Each prediction functions call the one of svm_predict functions.

    (Original SVM: use svm_predict(). It is the same with the libsvm library.)

    No encryption: use svm_predict2(). It is use encoding to compare with the IPFE version.

    IPFE on CPU: use svm_predict3(). It can select to use AVX2 or not by AVX2 definition in param.h file.

    The naive version IPFE on GPU: use svm_predict4(). 

    The parallel version of IPFE on GPU: use svm_predict5(). There are two kinds of key generation and decryption functions. "_gui2" is merged version and "_gui3" is no merged version.
    
    * You may also comment out one of the prediction functions for testing.

3) It can be tested as follow commend.

    $ ulimit -s unlimited

    $ ./svm_ipfe-gpu (data file) (model file) (output file)
