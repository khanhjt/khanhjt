# Attention is all you need
## 1. Encoder & Decoder
Máy tính chỉ có thể học dữ liệu nếu lập trình viên mã hóa các dữ liệu đó thành các số, sau khi trải qua nhiều các bước thì tiếp tục giải mã.
Quá trình đó gọi là Encoder và Decode. Vậy E và D là gì?
- **Encoder**: Chuyển input thành các features learning có khả năng học tập. Với các NN thì encoder này là các lớp ẩn, còn CNN thì là chuỗi các layers Conv + Maxpooling, RNN thì nó lại là các layers Embedding và Recurrent Neral Network.
- **Decoder**: Đầu ra của Encoder là đầu vào của Decoder, mục đích là tìm ra phân phối xác xuất từ các features learning ở Encoder để xác định nhãn. Kết quả nếu là các model phân loại là một nhãn, còn model seq2seq là một chuỗi các nhãn theo thứ tự.
 ![](https://phamdinhkhanh.github.io/assets/images/20190616_attention/pic1.png)
                *Mô hình seq2seq khi chưa có lớp attention*
 ![](https://phamdinhkhanh.github.io/assets/images/20190616_attention/pic2.png)

*Có lớp Attention*

Như trong hình có lớp Attention, từ 'I' trong tiếng Pháp là 'Je', đó đó lớp attention điều chỉnh một trọng số $\alpha$ lớn hơn ở context vector so với các từ khác.
Các phần màu xanh là các hidden state $h_i$ được trả ra ở mỗi phần. Context vector là tổ hợp tuyến tính của các output theo trọng số attention. Ở vị trí đầu của giai đoạn decoder thì context vector sẽ phân bố trọng số attention cao hơn so với các vị trí còn lại. Nghĩa là context vector tại mỗi time step sẽ được ưu tiên bằng cách đánh trọng số cao hơn cho các từ ở cùng vị trí time step. Ưu điêm khi sử dụng attention là mô hình lấy được toàn bộ bối cảnh của câu thay vì chỉ một từ input so với model thông thường.

**Các bước thực hiện**

1\. Đầu tiên tại time step thứ $t$ ta tính ra list các điểm số, mỗi điểm tương ứng với một cặp vị trí input $t$ và các vị trí còn lại theo công thức bên dưới:

$$score(h_t, \bar{h_s})$$

Ở đây $h_t$ cố định tại time step $t$ và là hidden state của từ mục tiêu thứ $t$ ở phase decoder, $\bar{h_s}$ là hidden state của từ thứ $s$ trong phase encoder. Công thức để tính score có thể là `dot product` hoặc `cosine similarity` tùy vào lựa chọn.

2\. Các scores sau bước 1 chưa được chuẩn hóa. Để tạo thành một phân phối xác xuất chúng ta đi qua hàm softmax khi đó ta sẽ thu được các trọng số attention weight.

$$\alpha_{ts} = \frac{\text{exp}(score(h_t, \bar{h_s}))}{\sum_{s'=1}^{S}\text{exp}(score(h_t, \bar{h_{s'}}))}$$

$\alpha_{ts}$ là phân phối attention (attention weight) của các từ trong input tới các từ ở vị trí $t$ trong output hoặc target.

3\. Kết hợp vector phân phối xác xuất $\alpha_{ts}$ với các vector hidden state để thu được context vector.

$$c_t = \sum_{s'=1}^{S} \alpha_{ts}\bar{h_{s'}}$$

4\. Tính attention vector để decode ra từ tương ứng ở ngôn ngữ đích. Attention vector sẽ là kết hợp của context vector và các hidden state ở decoder. Theo cách này attention vector sẽ không chỉ được học từ chỉ hidden state ở unit cuối cùng như hình 1 mà còn được học từ toàn bộ các từ ở vị trí khác thông qua context vector. Công thức tính output cho hidden state cũng
tương tự như tính đầu ra cho `input gate layer` trong mạng RNN:

$$a_t = f(c_t, h_t) = tanh(\mathbf{W_c}[c_t, h_t])$$

Kí hiệu $[c_t, h_t]$ là phép concatenate 2 vector $c_t, h_t$ theo chiều dài. Giả sử $c_t \in \mathbb{R}^{c}$, $h_t \in \mathbb{R}^{h}$ thì vector $[c_t, h_t] \in \mathbb{R}^{c+h}$.
$\mathbf{W_c} \in \mathbb{R}^{a\times(c+h)}$ trong đó $a$ là độ dài của attention vector. Ma trận mà chúng ta cần huấn luyện chính là $\mathbf{W_c}$

## 2. Transformer và Seq2Seq
![](https://phamdinhkhanh.github.io/assets/images/20190616_attention/pic3.png)
***Kiến trúc Transformer***
- **Encoder**: Đầu ra của sub-layer là $LayerNorm(x+Sublayer(x))$ có số chiều là 512
- **Decoder**: Có thêm 1 Masker ở sub-layer đầu tiên. Layer này không gì khác so với multi-head self-attention layer ngoại trừ được điều chỉnh để không đưa các từ trong tương lai vào attention.

Positional Encoding : đưa thêm yếu tố thời gian vào mô hình để làm tăng độ chuân xác. Nó là phép cộng vector mã hóa vị trí với vector biểu diễn từ. Mã hóa dưới dạng [0,1] hoặc sử dụng *sin*, *cos*.

## 3. Cơ chế Attention
### 3.1 Scale dot product attention.
Là một cơ chế self-attention: mỗi từ có thể điều chỉnh trọng số của từ khác trong câu sao cho từ ở vị trí càng gần nó nhất thì trọng số càng lớn và càng xa thì càng nhỏ.
Sau khi đi qua lớp embedding ta có ma trận X là đầu vào của encoder và decoder. 

![image](https://github.com/khanhjt/Attention-Is-All-You-Need/assets/105477211/20cc02f0-d000-41f1-ac97-d29bd409439c)

Các **Wq**, **Wk**, **Wv** là hệ số mà model cần huấn luyện. Nhân với ma trận **X** thu đc ma trận **Q, K, V**.

**Q, K** tính toán phân phối score cho các cặp từ, **V** dựa trên phân phối score để tính vector phân phối xác suất.
-> mỗi từ đc gán bởi 3 vector **Q, K, V**

![image](https://github.com/khanhjt/Attention-Is-All-You-Need/assets/105477211/dd238b11-d115-4f44-93ea-840423fa47f0)

Để tính score cho mỗi cặp từ ta cần tính dot-production giữa **Q** và **K**, phép tính này nhằm tìm ra mối liên hệ trọng số của các cặp từ. 
Sau đó chuẩn hóa bằng hàm Softmax, tiếp tục nhân Softmax với **V**  để tìm ra attention vector.

![image](https://github.com/khanhjt/Attention-Is-All-You-Need/assets/105477211/52680df1-cc20-4c61-bcf8-20ef74ca1af7)

*Tính cho từ I*

![image](https://github.com/khanhjt/Attention-Is-All-You-Need/assets/105477211/7e44c132-129a-4fa7-b091-4f86c9d13396)

*Tính attention cho toàn bộ câu*

            $Attention(\mathbf{Q, K, V}) = softmax(\frac{\mathbf{QK^T})}{\sqrt{d_k}}\mathbf{V} \tag{1}$

