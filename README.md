<h1 style="text-align: center">METHOD EXPLANATION</h1>

# Bài toán: Template Matching
Github: [https://github.com/TCT2001/dpi_exercise_1](https://github.com/TCT2001/dpi_exercise_1)

### Đầu vào:

* Ảnh gốc
* Những ảnh mẫu cần tìm

### Đầu ra

* Tìm, đánh dấu những ảnh mẫu có trong ảnh gốc

# Tools

* Python
* Thư viện: opencv, imutils, numpy

# Phương pháp

* **Bước 1**: import những thư viện cần thiết

* **Bước 2**: Định nghĩa những hằng số được sử dụng như
    * SCALE_MIN: giá trị scale nhỏ nhất
    * SCALE_MAX: giá trị scale lớn nhất
    * SCALE_STEP_DIFF: bước nhảy khi duyệt scale
    * SCALE_NUM: số lượng scale dùng để tìm
    * ROTATE_STEP_DIFF: bước nhảy khi xoay ảnh mẫu
    * THRESHOLD: ngưỡng đánh giá template matching
    * RECTANGLE_THICKNESS: độ dày của đường bao quay vật thể xác định được
* **Bước 3**:
    * Đọc ảnh gốc (imread) và grayscale (cvtColor với cv2.COLOR_BGR2GRAY) ảnh gốc đồng thời giảm nhiễu (bilateralFilter)

```python
img_rgb = cv2.imread(source_image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
```

* **Bước 4**: Duyệt qua từng ảnh mẫu
    * Với mỗi ảnh mẫu đều đọc và giảm nhiễu cho ảnh mẫu
    * Với mỗi ảnh mẫu, để có độ chính xác cao, ta chạy (process) với nhiều mức scale và nhiều góc độ xoay ảnh khác nhau
      thông qua các hằng số được định nghĩa ở **Bước 2**
    * Hàm matchTemplate của opencv sẽ là hàm thực hiện matching ảnh gốc với ảnh mẫu

```python
# Duyệt qua các ảnh mẫu
for index, template in enumerate(template_list):
    img_rgb = process(template, img_rgb, img_gray, bgr_color_list[index])
```

```python
# Hàm process
def process(template_image_path, img_rgb, img_gray, color):
    template = cv2.imread(template_image_path, 0)  # Đọc ảnh mẫu,
    template = cv2.bilateralFilter(template, 11, 17, 17)  # Giảm nhiễu, làm mịn cho ảnh mẫu

    # Duyệt qua các scale
    for scale in np.linspace(SCALE_MIN, SCALE_MAX, SCALE_NUM)[::-1]:
        # Thay đổi kích thước ảnh theo scale
        resized = imutils.resize(template, width=int(template.shape[1] * scale))
        for angle in np.arange(0, 360, ROTATE_STEP_DIFF):  # Duyệt qua các độ xoay của ảnh
            rotated_img = imutils.rotate(resized, angle)  # Xoay ảnh theo rotate
            w, h = rotated_img.shape[::-1]  # Lấy thông tin width, height của mẫu
            res = cv2.matchTemplate(img_gray, rotated_img, cv2.TM_CCOEFF_NORMED)  # Thực hiện matching
            threshold = THRESHOLD  # Chọn ngưỡng, ngưỡng càng cao thì độ chính xác cằng cao với cv2.TM_CCOEFF_NORMED

            # Lưu lại tọa độ của các vùng matched với ảnh mẫu với độ chính xác lớn hơn hoặc bằng threshold
            loc = np.where(res >= threshold)

            # Với các tạo độ đã lấy được (loc) đánh dấu bằng mỗi vùng tọa độ đó bằng
            # 1 hình chữ nhật có độ dày canh là RECTANGLE_THICKNESS, và màu là color
            if loc:
                for pt in zip(
                        *loc[::-1]):
                    cv2.rectangle(img_rgb, pt, (pt[0] + w + 4, pt[1] + h + 4), color=color,
                                  thickness=RECTANGLE_THICKNESS)
    return img_rgb  # Trả về ảnh gốc được đánh dấu để tiếp túc với những ảnh mẫu tiếp theo
```

* **Bước 5**: Lấy, trình bày và lưu lại ảnh img_rgb cuối cùng sau khi tất cả các mẫu được duyệt

```python
cv2.imwrite('template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
```