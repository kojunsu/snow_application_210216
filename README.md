## 프로젝트 : 간단한 스노우 카메라 앱 만들기


### 본 프로젝트는 스노우 카메라 앱과 같이 동영상의 사람 얼굴을 인식하여   그 위에 라이언 이미지를 씌웁니다.

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/62390565/108248309-07cfee80-7197-11eb-99b1-e2c9a68f0ec6.gif)



### 1. 개발 환경

- OS :Mac OS
- 개발 IDE : Pycharm
- Python : Ver 3.9.1

### 2. 라이브러리 
```python
#터미널에서 라이브러리 설치 

pip install opencv-python  #이미지 처리
pip install dlib           #얼굴 인식
pip install numpy          #행렬 연산
```

### 3. 블로그
- [공부한 내용](https://rkaclfrns.tistory.com/65)

### 4. 출처 
- [무료 동영상 다운 사이트](https://www.pexels.com/search/videos/face)


- [빵형의 개발도상국 Github](https://github.com/kairess/face_detector)

- [빵형의 개발도상국 Youtube](https://www.youtube.com/watch?v=tpWVyJqehG4&t=148s)

### 5. 코드
```python
import cv2, dlib, sys
import numpy as np

scaler = 0.3

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load video
cap = cv2.VideoCapture('samples/girl.mp4')
# load overlay image
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img

face_roi = []
face_sizes = []

# loop
while True:
  # read frame buffer from video
  ret, img = cap.read()
  if not ret:
    break

  # resize frame
  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  ori = img.copy()

  # find faces
  if len(face_roi) == 0:
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    # cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # no faces
  if len(faces) == 0:
    print('no faces!')

  # find facial landmarks
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face)
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    else:
      dlib_shape = predictor(roi_img, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

    for s in shape_2d:
      cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # compute face boundaries
    min_coords = np.min(shape_2d, axis=0)
    max_coords = np.max(shape_2d, axis=0)

    # draw min, max coords
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # compute face size
    face_size = max(max_coords - min_coords)
    face_sizes.append(face_size)
    if len(face_sizes) > 10:
      del face_sizes[0]
    mean_face_size = int(np.mean(face_sizes) * 1.8)

    # compute face roi
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    # draw overlay on face
    result = overlay_transparent(ori, overlay, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))

  # visualize
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img)
  cv2.imshow('result', result)
  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)
    ```
