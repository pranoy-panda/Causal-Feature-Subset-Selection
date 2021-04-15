# Instance-wise Causal Feature Selection for Model Interpretation

<img src="results/mnist.png" width="500" height="150">
<img src="results/8_2.png" width="500" height="150">
(Copper color indicates the 10% selected pixels for explaining why the black-box model predicted the image to be 8)

### Abstract
We  formulate  a  causal  extension  to  the  recently  intro-duced  paradigm  of  instance-wise  feature  selection  to  ex-plain  black-box  visual  classifiers.  Our  method  selects  asubset of input features that has the greatest causal effecton the model’s output.  We quantify the causal influence ofa subset of features by the Relative Entropy Distance mea-sure.   Under certain assumptions this is equivalent to theconditional mutual information between the selected subsetand the output variable. The resulting causal selections aresparser and cover salient objects in the scene. We show theefficacy of our approach on multiple vision datasets by mea-suring the post-hoc accuracy and Average Causal Effect ofselected features on the model’s output.

Authors: Pranoy Panda, Sai Srinivas Kancheti, Vineeth N Balasubramanian
Workshop: Causality in Vision Workshop, CVPR 2021
