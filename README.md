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

