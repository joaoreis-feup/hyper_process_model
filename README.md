# Zero-Shot Learning (ZSL) - Hyper Process Model


This repository contains the full implementation of the algorithm Hyper-Process Model (HPM) using as example a set of curves from Beta Distribution. This algorithm is used in the conference publication: 

"Reis, J., Gonçalves, G., & Link, N. (2017, October). Meta-process modeling methodology for process model generation in intelligent manufacturing. In Industrial Electronics Society, IECON 2017-43rd Annual Conference of the IEEE (pp. 3396-3402). IEEE."

where a private dataset from an industrial company is used. In the referred publication, the algorithm is not framed into a Zero-Shot Learning (ZSL) problem, and more specifically in the regression setting for ZSL.

# How to run the algorithm?

In the beginning of the code, there's a variable "HPM = True" that defines if the algorithm used is the HPM or Hyper-Model according to the the following publication:

"Jürgen Pollak and Norbert Link. From models to hyper-models of physical objects and industrial processes. In Electronics and Telecommunications (ISETC), 2016 12th IEEE International Symposium on, pages 317–320. IEEE, 2016."

"Jürgen Pollak, Alireza Sarveniazi, and Norbert Link. Retrieval of process methods from task descriptions and generalized data representations. The International Journal of Advanced Manufacturing Technology, 53(5-8):829–840, 2011."

The program writes all the results in a .txt file where a set of hyperparameters are tested. For the HPM, these hyperparameters are the Number of Components used in the decomposition of shapes (SSM) and the polynomial degree for the hyper-model training. As for the Hyper-Model alone (not framed in the HPM), these hyperparameters are the polynomial degree to train all source models and the polynomial degree for the hyper-model training. 

For more details about the implementation, please read the afore mentioned publication from Reis, J., Gonçalves, G., & Link, N..
