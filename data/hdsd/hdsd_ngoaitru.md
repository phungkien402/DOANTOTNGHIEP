# HDSD: Hướng dẫn sử dụng Module Quản lý khám bệnh ngoại trú
# Phần mềm quản lý bệnh viện thông minh EHC — E-healthcare Vietnam

---

## Giới thiệu

Tài liệu này là hướng dẫn sử dụng cho module Quản lý khám bệnh ngoại trú của phần mềm EHC (ehcHIS), gồm các chức năng:

- Màn hình hiển thị bệnh nhân chờ khám
- Phát loa cho bệnh nhân vào khám
- Nhập thông tin khám bệnh, hỏi bệnh
- Chỉ định cận lâm sàng (xét nghiệm, CĐHA)
- Chỉ định các dịch vụ Phẫu thuật Thủ thuật (PTTT)
- Tạo các mẫu (bộ) xét nghiệm, CĐHA
- Xem và in kết quả cận lâm sàng
- Nhập chẩn đoán ban đầu, bệnh chính, bệnh kèm theo ICD10
- Kê đơn thuốc BHYT, đơn thuốc mua ngoài, in đơn thuốc
- Đơn thuốc mẫu, đơn thuốc cũ
- Nhập xử trí (kết thúc khám, điều trị ngoại trú, nhập viện, chuyển tuyến)
- In tóm tắt bệnh án (thay cho sổ khám bệnh)
- Hẹn lịch khám lại
- Quản lý phòng lưu bệnh nhân
- Quản lý tủ trực của các phòng khám

---

## Màn hình hiển thị bệnh nhân chờ khám

Vào module **Khám bệnh** → Chọn **Màn hình chờ** → Chọn phòng khám muốn hiển thị → Nhấn **Lưu** để hoàn tất.

- Nhấn **Alt + Tab** để quay lại màn hình khám bệnh
- Đang ở giao diện màn hình chờ nhấn **Alt + F4** để tắt màn hình chờ

---

## Phát loa gọi bệnh nhân vào khám

Vào module **Khám bệnh** → Chọn bệnh nhân cần gọi.

Bảng màu trạng thái bệnh nhân:
- Không có mã màu trước số thứ tự: bệnh nhân chờ khám
- **Màu vàng**: đang khám
- **Màu xanh lá**: đã có đủ kết quả xét nghiệm, CĐHA
- **Màu xanh da trời có tích trắng**: bệnh nhân đã xử trí xong
- **Màu vàng viền xanh**: đang mở lại bệnh án
- **Loa màu xanh**: đã gọi bệnh nhân
- **Loa màu đỏ**: đã gọi bệnh nhân khám lại

Nhấn **Gọi khám** hoặc **F9** để gọi bệnh nhân lần đầu; nhấn **Gọi KL** hoặc **F10** để gọi lại. Có thể chuột phải vào tên bệnh nhân → Chọn **Gọi bệnh nhân**.

Nhấn **Bắt đầu khám** hoặc **F1** để bắt đầu khám.

---

## Nhập thông tin khám bệnh, hỏi bệnh

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn **Khám bệnh** → Nhập thông tin vào các ô trắng tương ứng.

- Những chữ có gạch chân màu xanh có thể nhấn để chọn kết quả xét nghiệm, CĐHA làm căn cứ khám bệnh
- Những ô màu xám có biểu tượng hình cần nhấn vào hình để hiện ra bảng chọn

Nhấn **Lưu** hoặc **Ctrl + S** để hoàn tất. Nếu nút lưu tự xám lại là phần mềm đã tự động lưu, người dùng không cần thao tác lưu.

---

## Chỉ định cận lâm sàng (xét nghiệm, CĐHA)

Vào module **Khám bệnh** → Chọn bệnh nhân → Nhấn **Dịch vụ** (bảng dịch vụ hiển lên) → Tích chọn dịch vụ cần thực hiện.

- Những dịch vụ đã được chỉ định sẽ hiện ở bảng nhỏ bên dưới với hình ngôi sao màu xanh
- Nhấn **Ctrl + F** để tìm nhanh dịch vụ
- Hoặc tìm theo cây thư mục đã được phân loại ở bảng bên trái

Nhấn **Lưu** để hoàn tất. Chọn **Lưu + In** để lưu đồng thời hiển bảng in phiếu chỉ định dịch vụ.

---

## Chỉ định dịch vụ Phẫu thuật Thủ thuật (PTTT)

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn **Dịch Vụ** → Kéo bảng danh sách dịch vụ bên trái xuống dưới tới cây thư mục phân loại **Chuyên Khoa**.

- Tìm theo cây thư mục đã được phân loại ở bảng bên trái
- Hoặc nhấn **Ctrl + F** để tìm nhanh dịch vụ

Tích chọn dịch vụ cần thực hiện → Nhấn **Lưu** để hoàn tất.

---

## Tạo mẫu (bộ) xét nghiệm, CĐHA

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn **Dịch vụ** → Chỉ định những dịch vụ cần tạo mẫu → Chọn **Mẫu chỉ định** → **Lưu vào mẫu mới**.

Bảng tên mẫu hiện ra:
- Nhập tên mẫu (bộ) xét nghiệm, CĐHA
- Nhấn **Lưu** để hoàn tất

Nhấn **Lưu** hoặc thoát bảng Chỉ định dịch vụ.

Lần chỉ định sau khi nhấn vào **Mẫu chỉ định**, mẫu đã lưu sẽ xuất hiện ở mục Mẫu chỉ định riêng với tên đã đặt.

---

## Xem và in kết quả cận lâm sàng

Vào module **Khám bệnh** → Chọn bệnh nhân đã chỉ định dịch vụ → Chọn form **Xét nghiệm** hoặc **CĐHA** (ở cùng dòng) → Chọn phiếu dịch vụ muốn xem kết quả.

- Kết quả sẽ hiện ra
- Nhấn vào phiếu in kết quả hoặc nhấn **F3** để in phiếu kết quả
- Nhấn **In** hoặc **Ctrl + P** để in

---

## Nhập chẩn đoán ban đầu, bệnh chính, bệnh kèm theo ICD10

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn form **Khám bệnh** → Nhập vào ô **Chẩn đoán bệnh ban đầu** mã ICD10 hoặc tên bệnh (phần mềm sẽ gợi ý mã ICD10 và tên bệnh).

Chọn hình **Bệnh kèm theo** (bảng Chẩn đoán bệnh hiện ra):
- Nhập mã bệnh hoặc tên bệnh → Nhấn **Enter** để bệnh kèm theo xuất hiện ở bảng dưới
- Nhấn **Lưu** để hoàn tất

Nhấn **Lưu** hoặc **Ctrl + S** để hoàn tất. Nếu nút lưu tự xám lại là phần mềm đã tự động lưu.

---

## Kê đơn thuốc BHYT, đơn thuốc mua ngoài, in đơn thuốc

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn **Thuốc/VT** → **Chỉ định thuốc** (hoặc **Ctrl + T**).

Bảng chỉ định thuốc hiện ra, chọn kho thuốc muốn kê:
- **Kho Nhà thuốc bệnh viện**: đơn thuốc mua ngoài
- **Kho BHYT**: đơn thuốc bảo hiểm
- Ngoài ra có thể kê từ những kho được phân quyền

Nhập tên thuốc, số lượng, hướng dẫn sử dụng → Nhấn **OK** hoặc **Ctrl + A** để lưu thuốc vào danh sách → Nhấn **Cấp toa cho về** hoặc **Lưu (Ctrl + S)** để hoàn tất.

**Để in đơn thuốc** (sau khi đã lưu, bảng chỉ định tự đóng):
1. Chọn module Khám bệnh → Chọn bệnh nhân
2. Chọn form **Thuốc**
3. Chọn phiếu **Chỉ định thuốc** cần xem, in
4. Nhấn **In đơn thuốc** hoặc **F3** để in
5. Bảng In hiện ra → Nhấn **In** hoặc **Ctrl + P** để in

---

## Đơn thuốc mẫu, đơn thuốc cũ

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn **Thuốc/VT** (hoặc **Ctrl + T**) → Chỉ định thuốc → Chọn **Đơn thuốc mẫu** → **Lưu vào mẫu mới**.

Bảng tên mẫu hiện ra:
- Nhập thông tin nhóm đơn mẫu, tên mẫu
- Nhấn **Lưu** để hoàn tất

Chọn **Lưu** hoặc đóng bảng chỉ định.

Lần chỉ định thuốc sau: Chọn **Đơn thuốc mẫu** → Đơn thuốc mẫu đã lưu xuất hiện ở mục **Tổng hợp** với tên đã đặt.

---

## Nhập xử trí (kết thúc khám, điều trị ngoại trú, nhập viện, chuyển tuyến)

Vào module **Khám bệnh** → Chọn bệnh nhân → Chọn form **Xử trí** → Chọn **Kết quả điều trị** → Chọn **Hình thức xử trí**.

Tùy vào hình thức xử trí, bảng tương ứng sẽ hiện ra:
- **Ra viện**: hiện ra bảng Cập nhật thông tin ra viện
- **Chuyển phòng khám**: hiện ra bảng Chuyển phòng khám
- **Chuyển tuyến**: hiện ra bảng Cập nhật thông tin chuyển tuyến
- **Nhập viện**: hiện ra ô để nhập/chọn khoa nội trú cho bệnh nhân
- Tất cả bảng đều cần nhấn **Lưu** để hoàn tất

Nhấn **Xử trí** để hoàn tất.

Sau khi xử trí, chữ ở form Xử trí và Khám bệnh sẽ chuyển sang màu xanh và không thể sửa. Nếu cần sửa thì phải mở lại bệnh án.

---

## In tóm tắt bệnh án (thay cho sổ khám bệnh)

Dành cho bệnh nhân khám bệnh ngoại trú: Vào module **Khám bệnh** → Chọn **In** → Chọn **Bệnh án khám bệnh** → Bảng in Bệnh án khám bệnh hiện ra → Chọn **In** hoặc nhấn **Ctrl + P** để in.

---

## Hẹn lịch khám lại

Vào module **Khám bệnh** → Chọn bệnh nhân → Chuột phải vào tên bệnh nhân → Chọn **Đặt hẹn**.

Bảng đặt lịch hẹn hiện ra → Nhập thông tin lịch hẹn (có thể bổ sung thêm thông tin bệnh nhân nếu chưa đầy đủ) → Nhấn **Lưu lại** hoặc **F2** để hoàn tất.

---

## Quản lý phòng lưu bệnh nhân

Phòng lưu bệnh nhân là phòng thuộc khoa khám bệnh, cần cấu hình trong cài đặt hệ thống. Phòng lưu có thể được chỉ định từ module Đón tiếp hoặc chuyển phòng trong module Khám bệnh.

### Chọn phòng lưu trong module Đón tiếp

1. Chọn module **Đón tiếp**
2. Nhấn **Gọi số tiếp theo** để gọi bệnh nhân
3. Nhập thông tin của bệnh nhân
4. Chọn **Dịch vụ khám** cho bệnh nhân
5. Chọn **Phòng khám** cho bệnh nhân
6. Chọn **Phòng lưu bệnh nhân** nếu muốn chuyển bệnh nhân vào phòng lưu
   - Có thể chọn nhiều phòng bằng cách nhấn **Khám nhiều phòng** → Chọn phòng khám mong muốn → Nhấn **Lưu**
7. Nhấn **Lưu lại** hoặc **F2** để hoàn tất đăng ký khám

### Chuyển phòng lưu trong module Khám bệnh

1. Chọn module **Khám bệnh** → Chọn bệnh nhân muốn chuyển phòng
2. Chọn **Xử trí** → Chọn **Kết quả điều trị**
3. Chọn hình thức xử trí **Chuyển phòng khám**
4. Nhập thông tin đầy đủ: Lý do, Yêu cầu khám, Phòng khám
5. Nhấn **Lưu** để hoàn tất
6. Nhấn **Xử trí** để hoàn tất chuyển phòng

### Cấu hình phòng lưu bệnh nhân

Chọn **Hệ thống** → **Quản lý hệ thống** → **Danh sách phòng** → Nhấn **Thêm** để tạo phòng mới → Nhập thông tin cấu hình → Nhấn **Lưu** để hoàn tất.

---

## Quản lý tủ trực của phòng khám

Tủ trực của các phòng khám được quản lý tại 2 module: **Tủ trực thuốc** và **Tủ trực vật tư**.

1. Nhấn vào **mũi tên nhỏ** cạnh tên tủ trực để hiện ra danh sách tủ trực
2. Nhấn **Tạo phiếu** để hiện ra những thao tác quản lý tủ trực → Chọn thao tác mong muốn

Ví dụ: Chọn **Bổ sung cơ số tủ trực** (bảng Bổ sung cơ số tủ trực hiện ra) → Nhập thông tin thuốc, số lượng → Nhấn **OK** hoặc **Ctrl + A** để lưu vào danh sách → Chọn thao tác tương ứng để kết thúc.

Trạng thái phiếu tủ trực:
- **Bản nháp**: sau khi nhấn Lưu Phiếu
- **Phiếu gửi đi**: đã được Kho duyệt
- **Đã nhập Kho tủ trực từ kho tổng**: hoàn tất
- **Đã nhập kho từ nhà cung cấp**: thành công
