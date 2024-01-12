# YOLO
## 1. Kiến trúc mạng YOLO
kiến trúc YOLO bao gồm: base network là các mạng convolution làm nhiệm vụ trích xuất đặc trưng. Phần phía sau là những Extra Layers được áp dụng để phát hiện vật thể trên feature map của base network.

base network của YOLO sử dụng chủ yếu là các convolutional layer và các fully conntected layer. Các kiến trúc YOLO cũng khá đa dạng và có thể tùy biến thành các version cho nhiều input shape khác nhau.
![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/8ca53097-0486-4f17-915e-7b778a7c3603)
Thành phần Darknet Architechture được gọi là base network có tác dụng trích suất đặc trưng. Output của base network là một feature map có kích thước 7x7x1024 sẽ được sử dụng làm input cho các Extra layers có tác dụng dự đoán nhãn và tọa độ bounding box của vật thể.
YOLO đang hỗ trợ 2 đầu vào chính là 416x416 và 608x608
ảnh khi được đưa vào mô hình sẽ được scale để về chung một kích thước phù hợp với input shape của mô hình và sau đó được gom lại thành batch đưa vào huấn luyện.
Kích thước của feature map sẽ phụ thuộc vào đầu vào. Đối với input 416x416 thì feature map có các kích thước là 13x13, 26x26 và 52x52. Và khi input là 608x608 sẽ tạo ra feature map 19x19, 38x38, 72x72.
## 2. Output
Output của mô hình YOLO là một véc tơ sẽ bao gồm các thành phần:
      $$y^{T} = [p_0,\langle\underbrace{t_x, t_y, t_w, t_h}_{\text{bounding box}}\rangle, \langle\underbrace{p_1, p_2,..., p_c}_{\text{scores of c classes}}\rangle]$$
       * p0  là xác suất dự báo vật thể xuất hiện trong bounding box.
       * $$ \langle\underbrace{t_x, t_y, t_w, t_h}_{\text{bounding box}}\rangle$$   giúp xác định bounding box. Trong đó tx, ty là tọa độ tâm và tw, th là kích thước rộng, dài của bounding box.
       * $$ \langle\underbrace{p_1, p_2,…, p_c}_{\text{scores of c classes}}\rangle$$  là véc tơ phân phối xác suất dự báo của các classes.
output sẽ được xác định theo số lượng classes theo công thức: $$ (\text{n_class}+5)$$

# 4. Dự báo trên nhiều feature map
![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/108e136b-56c9-4303-a134-090e5a4556b7)
Các feature maps của mạng YOLOv3 với input shape là 416x416, output là 3 feature maps có kích thước lần lượt là 13x13, 26x26 và 52x52.
Trên mỗi một cell của các feature map chúng ta sẽ áp dụng 3 anchor box để dự đoán vật thể. Như vậy số lượng các anchor box khác nhau trong một mô hình YOLO sẽ là 9 (3 featue map x 3 anchor box).
Đồng thời trên một feature map hình vuông S x S, mô hình YOLOv3 sinh ra một số lượng anchor box là: S x S x 3. Như vậy số lượng anchor boxes trên một bức ảnh sẽ là:

      $$(13 \times 13 + 26 \times 26 + 52 \times 52) \times 3 = 10647 \text{(anchor boxes)}$$
  
số lượng rất lớn và là nguyên nhân khiến quá trình huấn luyện mô hình YOLO vô cùng chậm bởi chúng ta cần dự báo đồng thời nhãn và bounding box trên đồng thời 10647 bounding boxes.

#Lưu ý khi huấn luyện YOLO:

*Khi huấn luyện YOLO sẽ cần phải có RAM dung lượng lớn hơn để save được 10647 bounding boxes như trong kiến trúc này.
*Không thể thiết lập các batch_size quá lớn như trong các mô hình classification vì rất dễ Out of memory. Package darknet của YOLO đã chia nhỏ một batch thành các subdivisions cho vừa với RAM.
*Thời gian xử lý của một step trên YOLO lâu hơn rất rất nhiều lần so với các mô hình classification. Do đó nên thiết lập steps giới hạn huấn luyện cho YOLO nhỏ. Đối với các tác vụ nhận diện dưới 5 classes, dưới 5000 steps là có thể thu được nghiệm tạm chấp nhận được. Các mô hình có nhiều classes hơn có thể tăng số lượng steps theo cấp số nhân tùy bạn.



## Anchor box
Để tìm được bounding box cho vật thể, YOLO sẽ cần các anchor box làm cơ sở ước lượng. Những anchor box này sẽ được xác định trước và sẽ bao quanh vật thể một cách tương đối chính xác. Sau này thuật toán regression bounding box sẽ tinh chỉnh lại anchor box để tạo ra bounding box dự đoán cho vật thể. Trong một mô hình YOLO:
* Mỗi một vật thể trong hình ảnh huấn luyện được phân bố về một anchor box. Trong trường hợp có từ 2 anchor boxes trở lên cùng bao quanh vật thể thì ta sẽ xác định anchor box mà có IoU với ground truth bounding box là cao nhất.
* Mỗi một vật thể trong hình ảnh huấn luyện được phân bố về một cell trên feature map mà chứa điểm mid point của vật thể. Chẳng hạn như hình chú chó trong hình 3 sẽ được phân về cho cell màu đỏ vì điểm mid point của ảnh chú chó rơi vào đúng cell này. Từ cell ta sẽ xác định các anchor boxes bao quanh hình ảnh chú chó.
  ![image](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/2a4293dd-8b65-4c0b-9ea5-a1c15895bdeb)
# Hàm loss
     ![Screenshot from 2024-01-02 10-14-08](https://github.com/khanhjt/Doc_Nautilus/assets/105477211/2cb45e0f-d36e-4469-b852-73a8443362ba)


Trong đó: 
* $$\mathbb{1}_i^\text{obj}$$ hàm indicator có giá trị 0,1 nhằm xác định xem cell i có chứa vật thể hay không. Bằng 1 nếu chứa vật thể và 0 nếu không chứa.
* $$\mathbb{1}_{ij}^\text{obj}$$ Cho biết bounding box thứ j của cell i có phải là bouding box của vật thể được dự đoán hay không?
* $$C_{ij}$$ Điểm tin cậy của ô i , P(contain object) * IoU (predict bbox, ground truth bbox).
* $$\hat{C}_{ij}$$ : Điểm tự tin dự đoán.
* C la tập hợp tất cả các lớp
* $$\mathcal{L}_\text{loc}$$  là hàm mất mát của bounding box dự báo so với thực tế.
* $$\mathcal{L}_\text{cls}$$  là hàm mất mát của phân phối xác suất. Trong đó tổng đầu tiên là mất mát của dự đoán có vật thể trong cell hay không? Và tổng thứ 2 là mất mát của phân phối xác suất nếu có vật thể trong cell.
