# Segmentation 완전정복 Crew

- 빌드 페이지 : https://pseudo-lab.github.io/SegCrew-Book
- reference1: https://github.com/Pseudo-Lab/Jupyter-Book-Template
- reference2: https://github.com/Pseudo-Lab/pytorch-guide

## jupyter book build

- init
  - 페이지 작성은 `.md`, `.ipynb` 형식으로 작성

- git clone 

  - ```
    git clone https://github.com/Pseudo-Lab/SegCrew-Book.git
    ```

- 페이지 작성 파일 이동

  - `SegCrew-Book/book/docs` 에 위치시킬 것
  - `ch1` 폴더 내에 작성

- `_toc.yml` 변경

  - `SegCrew-Book/book` 내 `_toc.yml` 파일 변경

  - ```yaml
	format: jb-book
	root: docs/index
	chapters:
	- file: docs/ch0/00 Survey
	  sections:
	  - file: docs/ch0/Survey-Semantic
	  - file: docs/ch0/Survey-Instance
	  - file: docs/ch0/Survey-Panoptic
	  - file: docs/ch0/Survey-WeaklySemi
	- file: docs/ch1/01 Semantic Segmentation
	  sections:
	  - file: docs/ch1/fcn
	- file: docs/ch2/02 Instance Segmentation
	- file: docs/ch3/03 Panoptic Segmentation
	- file: docs/ch4/04 Weakly-supervised Segmentation
	- file: docs/ch5/05 Semi-supervised Segmentation
    ```

  - 위 코드 참조하여 추가한 페이지 이름 변경

- Jupyter book 설치

  - ```
    pip install -U jupyter-book
    ```

- 폴더 이동

  - ```
    cd pytorch-guide
    ```

- (로컬) Jupyter book build

  - ```
    jupyter-book build book/
    ```

  - cmd 창 내 `Or paste this line directly into your browser bar:` 이하의 링크를 사용하면 로컬에서 jupyter book 을 빌드할 수 있음

- (온라인) Jupyter book build

  - 변경 내용 push 할 것

  - ```python
    pip install ghp-import
    ghp-import -n -p -f book/_build/html -m "20-08-09 publishing"
    ```

  - https://pseudo-lab.github.io/pytorch-guide/ 링크 접속

## Writing Rules

1. 제목은 리뷰하고자 하는 논문의 nickname또는 논문 제목과 투고된 학회/학술지 명을 다음과 같은 양식으로 기재한다. 

```plaintext
{"NickName or 제목"} - {"학회/학술지 명"}
```

2. 본문 최상단에 논문 정보를 기재한다. 
- Jupyter-book의 "admonition" block에 다음과 같은 양식으로 기입한다. 

```plaintext
```{admonition} Information
- **Title:** {논문 제목}, {학회/학술지명}

- **Reference**
    - Paper: {논문 링크}
    - Code: {Official code}
    - Review: {OpenReview 등 정식 리뷰 결과 링크}
    
- **Review By:** {리뷰 작성자 기입}

- **Edited by:** {리뷰 편집자 기입}

- **Last updated on {최종 update 날짜 e.g. Jan. 5, 2022}**
```
```
3. 본문에 그림 추가 시 출처를 기입한다. 
- 
```plaintext
:::{figure-md} {figure-tag}
<img src="{주소}" alt="{tag명}" class="bg-primary mb-1" width="{크기, 800px이 기준}">

{제목} \  (source: {출처 기입, 아카이브 링크 기준으로 기입하는 것을 기준으로 함. e.g. arXiv:1803.10464})
:::
```
- 논문의 표를 추가하는 경우 다음 양식을 활용한다. 
```plaintext
 ```{image} pic/affinitynet/aff9.png
:alt: aff9.png
:class: bg-primary mb-1
:align: center
```
```