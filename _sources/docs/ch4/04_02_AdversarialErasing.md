# B. Adversarial Erasing

- CAM은 Classification을 위한 network에서 얻어지기 때문에 두드러진 특징(intra-category variations)을 가진 area에 의존한다. 즉 discriminative area에 부분에 정보가 집중된다. 

- 이런 문제점을 해결하기 위해 정보가 집중되는 영역을 반복적으로 제거하여 CAM을 추출하여 object 전체에 대해 attention을 가지도록 학습하는 방법을 adversarial erasing 기법이라고 한다. 

:::{figure-md} markdown-fig
<img src="pic/seenet/seenet2.png" alt="seenet2" class="bg-primary mb-1" width="800px">

AE-PSL architecture (Source: arXiv:1703.08448)
:::

- 하지만 반복적으로 attention을 확장하게되면, object가 아닌 영역에까지 attention이 확장되어 정확도가 감소하는 문제가 발생한다. 

- 본 chapter에서는 Adversarial Erasing를 이용한 weakly supervised segmentation 연구 결과를 리뷰한다.

    (1) AE-PSL - CVPR 2017
    
    (2) SeeNet - NeurIPS 2018