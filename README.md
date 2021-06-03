Implementation of "Privacy-Preserving Fair Machine Learning Without Collecting Sensitive Demographic Data" (IJCNN 2021).

### Abstract
With the rising concerns over privacy and fairness in machine learning, privacy-preserving fair machine learning has received tremendous attention in recent years. However, most existing fair models still need to collect sensitive demographic data, which may be impossible given privacy regulations. To address the dilemma between model fairness and sensitive data collection, we propose DicPF, a distributed and privacy-preserving fair learning framework that operates without collecting sensitive demographic data. In particular, DicPF assumes multiple local agents and a modeler are distributed, and sensitive demographic data is separately held by multiple local agents. To assist fair learning at the modeler, each agent learns a fair local dictionary and sends it to the modeler. The modeler learns a fair model based on an aggregated dictionary. Under DicPF framework, we propose a private z-Sparse Fair Learner. Extensive experiments on three real-world datasets demonstrate the efficiency of the proposed model. Compared with the state-of-the-art fair learners, the proposed z-Sparse Fair Learner achieves superior fairness performance by lowering prediction disparity. We also develop a privacy inference model to demonstrate the excellent privacy-preserving performance of DicPF. Finally, we theoretically analyze z-Sparse Fair Learner and prove upper bounds on its model fairness and accuracy.

### Citation
Update later

