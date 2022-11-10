# A. Learning Pixel Affinity

- Weakly supervised learning을 이용한 semantic segmantation을 위해 주로 사용되는 CAM의 경우 discriminate한 부분에 정보가 집중되어 sparse하고 blurrly한 정보를 포함하는 단점이 존재한다.
 
- Learning Pixel Affinity 분류의 연구들은 이러한 CAM의 단점을 극복하기 위해  의미론적 유사도 (semantic affinities)를 이용하며, CAM의 전/후처리 및 affinity를 계산하기 위한 network를 학습하는 방법에 따라 다양한 방법들이 제안되고 있다. 


:::{figure-md} markdown-fig
<img src="pic/weakly5.png" alt="weakly5" class="bg-primary mb-1" width="800px">

Overview of Affinity based weakly-supervised semantation 
:::

- 기본적으로 Object를 구성하는 pixel을 class label을 가져야 하며, 같은 semantic을 가질 가능성이 높다는 점을 이용하여, CAM에서 추출한 두 location이 같은 class label을 가지는지 판단할 수 있는 network를 학습하는 것을 목표로 한다. 

- CAM과 각 pixel location간의 Affinity를 이용하여 Pseudo Label을 생성하고, 이를 이용하여 segmentation을 수행하는 network를 학습한다. 

    :::{note}
    Pseudo label을 만드는 과정은 매우 복잡하고 연산량이 많으므로, pseudo label을 이용하여 network를 학습하는게 test time에서 효율적이며, network 학습을 과정에서 pseudo label의 noise에 강인한 모델을 학습할 수 있다. 또한 보통 pseudo label을 바로 적용하는 것 보다 성능이 더 좋다고 알려져 있다.
    :::

- 본 chapter에서는 pixel affinity를 이용한 weakly supervised segmentation 연구 결과를 리뷰한다. 

    - [Learning Pixel-level Semantic Affinity Image-level Labels for Weakly Supervised Semantic Segmentation, CVPR 2018](https://pseudo-lab.github.io/SegCrew-Book/docs/ch4/04_01_01_AffinityNet.html)
    
    - [Weakly-Supervised Semantic Segmentation by Iterative Affinity Learning, IJCV 2020](https://pseudo-lab.github.io/SegCrew-Book/docs/ch4/04_01_02_IAL.html)