import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Vẫn dùng seaborn để vẽ heatmap cho ma trận nhầm lẫn custom

# Bước 1: Import các lớp và hàm cần thiết từ module tự tạo
from custom_ml_algorithms import (
    CustomRandomForestClassifier,
    CustomDecisionTreeClassifier, # Cần thiết cho pickle nếu model lưu cây quyết định bên trong
    Node,                       # Cần thiết cho pickle
    custom_train_test_split,
    custom_accuracy_score,
    custom_classification_report_simplified,
    custom_confusion_matrix # Import hàm này nếu bạn đã thêm vào custom_ml_algorithms.py
)

# Đường dẫn tệp dữ liệu (giữ nguyên hoặc thay đổi nếu cần)
# DATA_PICKLE_PATH = '/kaggle/input/picketdataset/data.pickle' # Đường dẫn cũ
DATA_PICKLE_PATH = './data.pickle' # Sử dụng đường dẫn tương đối nếu data.pickle cùng thư mục

print(f"Đang tải dữ liệu từ '{DATA_PICKLE_PATH}'...")
try:
    with open(DATA_PICKLE_PATH, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp dữ liệu '{DATA_PICKLE_PATH}'.")
    exit()

data_list = data_dict['data']
labels_list = data_dict['labels']

# Xử lý dữ liệu đầu vào (tương tự như trong train_classifier.py đã sửa)
processed_data = []
processed_labels = []
expected_feature_length = 42

for i, sample in enumerate(data_list):
    sample_arr = np.array(sample)
    if len(sample_arr) == expected_feature_length:
        processed_data.append(sample_arr)
        processed_labels.append(labels_list[i])
    elif len(sample_arr) > expected_feature_length:
        processed_data.append(sample_arr[:expected_feature_length])
        processed_labels.append(labels_list[i])
    else:
        # print(f"Cảnh báo (evaluate): Mẫu {i+1} (nhãn: {labels_list[i]}) có độ dài {len(sample_arr)}, bị bỏ qua.")
        pass

if not processed_data:
    print("Lỗi (evaluate): Không có dữ liệu hợp lệ để đánh giá. Kết thúc.")
    exit()

data = np.asarray(processed_data)
labels = np.asarray(processed_labels)
print(f"Tổng số mẫu hợp lệ để đánh giá: {len(data)}")

# Chia dữ liệu bằng hàm tự custom
x_train, x_test, y_train, y_test = custom_train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)
print(f"Kích thước tập huấn luyện (evaluate): X_train {x_train.shape}, y_train {y_train.shape}")
print(f"Kích thước tập kiểm tra (evaluate): X_test {x_test.shape}, y_test {y_test.shape}")


# Huấn luyện một CustomRandomForestClassifier mới (giống như tệp evaluate.py gốc đã làm)
# Nếu bạn muốn đánh giá một model đã được huấn luyện và lưu trước đó (ví dụ: 'custom_random_forest_model.p'),
# bạn cần tải model đó ở đây thay vì huấn luyện lại.
# Ví dụ tải model đã lưu:
# with open('custom_random_forest_model.p', 'rb') as f_model:
#     model_data = pickle.load(f_model)
# model = model_data['model']
# print("Model tự custom đã huấn luyện trước được tải để đánh giá.")

# Hiện tại, theo logic gốc của evaluate.py, chúng ta huấn luyện lại:
print("\nBắt đầu huấn luyện mô hình Custom Random Forest để đánh giá...")
model = CustomRandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2)
model.fit(x_train, y_train)
print("Huấn luyện mô hình (evaluate) hoàn tất.")

# Dự đoán trên tập kiểm tra
y_pred = model.predict(x_test)

# Tính toán độ chính xác bằng hàm tự custom
score = custom_accuracy_score(y_test, y_pred)
print(f"\n{score*100:.2f}% mẫu trong tập kiểm tra (evaluate) được phân loại chính xác!")

# Lưu model vừa huấn luyện (tùy chọn, có thể đặt tên khác)
model_eval_filename = 'custom_model_evaluated.p'
with open(model_eval_filename, 'wb') as f:
    pickle.dump({'model': model}, f)
print(f"Mô hình vừa đánh giá đã được lưu vào '{model_eval_filename}'")

# In báo cáo phân loại (phiên bản đơn giản)
print("\nBáo cáo phân loại (evaluate - phiên bản đơn giản hóa):")
print(custom_classification_report_simplified(y_test, y_pred))

# Ma trận nhầm lẫn (sử dụng hàm custom_confusion_matrix)
# Phần này tùy chọn, yêu cầu bạn đã thêm hàm custom_confusion_matrix
if 'custom_confusion_matrix' in globals(): # Kiểm tra xem hàm có tồn tại không
    # Lấy danh sách các lớp duy nhất từ y_test và y_pred để đảm bảo thứ tự nhất quán
    # Hoặc tốt hơn là lấy từ model.classes_ nếu model của bạn lưu trữ nó
    unique_labels_for_cm = model.classes_ if hasattr(model, 'classes_') and model.classes_ is not None else sorted(np.unique(np.concatenate((y_test, y_pred))))

    cm, cm_labels = custom_confusion_matrix(y_test, y_pred, labels=unique_labels_for_cm)
    
    if cm is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=cm_labels, yticklabels=cm_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Custom Confusion Matrix')
        plt.show()
else:
    print("\nHàm custom_confusion_matrix không được tìm thấy. Bỏ qua vẽ ma trận nhầm lẫn.")


# Trực quan hóa Precision, Recall, F1-score (đơn giản, không giống sklearn hoàn toàn)
# Đoạn này cần dữ liệu chi tiết từ báo cáo phân loại, phiên bản đơn giản có thể không đủ.
# Bạn có thể sửa đổi custom_classification_report_simplified để trả về dict thay vì string
# hoặc phân tích string đó. Hiện tại, tôi sẽ bỏ qua phần vẽ biểu đồ chi tiết này
# vì custom_classification_report_simplified chỉ in ra string.
# Nếu bạn muốn vẽ, bạn cần sửa hàm report để trả về dictionary các metric.

print("\nĐánh giá hoàn tất.")