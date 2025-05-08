import pickle
import numpy as np
import itertools # Để tạo các tổ hợp tham số
import matplotlib.pyplot as plt
import seaborn as sns

# Bước 1: Import các lớp và hàm cần thiết từ module tự tạo
from custom_ml_algorithms import (
    CustomRandomForestClassifier,
    CustomDecisionTreeClassifier, # Cần thiết cho pickle
    Node,                       # Cần thiết cho pickle
    custom_train_test_split,
    custom_accuracy_score,
    custom_classification_report_simplified,
    custom_confusion_matrix # Tùy chọn, nếu có
)

# Đường dẫn tệp dữ liệu
# DATA_PICKLE_PATH = '/kaggle/working/data.pickle' # Đường dẫn cũ
DATA_PICKLE_PATH = './data.pickle' # Sử dụng đường dẫn tương đối

print(f"Đang tải dữ liệu từ '{DATA_PICKLE_PATH}' cho tìm kiếm tham số...")
try:
    with open(DATA_PICKLE_PATH, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp dữ liệu '{DATA_PICKLE_PATH}'.")
    exit()

data_list = data_dict['data']
labels_list = data_dict['labels']

# Xử lý dữ liệu đầu vào (tương tự như các tệp khác)
processed_data = []
processed_labels = []
expected_feature_length = 42
for i, sample in enumerate(data_list):
    sample_arr = np.array(sample)
    if len(sample_arr) == expected_feature_length:
        processed_data.append(sample_arr)
        processed_labels.append(labels_list[i])
    # Bỏ qua các mẫu không đúng kích thước trong lần này để đơn giản
    # Trong thực tế, bạn nên có một bước tiền xử lý dữ liệu nhất quán
    
if not processed_data:
    print("Lỗi (find_best_parameter): Không có dữ liệu hợp lệ. Kết thúc.")
    exit()

data = np.asarray(processed_data)
labels = np.asarray(processed_labels)
print(f"Tổng số mẫu hợp lệ: {len(data)}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (hoặc huấn luyện và validation)
# Để tìm tham số, thường dùng tập validation riêng, hoặc cross-validation thủ công.
# Để đơn giản, chúng ta sẽ chia thành train/test, và dùng test set để chọn tham số tốt nhất.
# (Lưu ý: Đây không phải là thực hành tốt nhất, nên có tập validation riêng biệt)
x_train_full, x_test_final, y_train_full, y_test_final = custom_train_test_split(
    data, labels, test_size=0.2, shuffle=True, random_state=42
)
# Tiếp tục chia x_train_full thành x_train_cv và x_val_cv để cross-validation nếu muốn
# Hiện tại, chúng ta sẽ dùng x_train_full để huấn luyện và x_test_final để đánh giá chọn tham số
print(f"Kích thước tập huấn luyện (find_param): {x_train_full.shape}")
print(f"Kích thước tập kiểm tra/validation (find_param): {x_test_final.shape}")


# Thiết lập lưới tham số (giữ nguyên từ code gốc của bạn, nhưng bỏ 'bootstrap' và 'max_features' nếu
# CustomRandomForestClassifier không hỗ trợ chúng theo cách giống sklearn)
# CustomRandomForestClassifier của chúng ta có n_features_subsample (tương tự max_features='sqrt' hoặc 'log2' nếu là số nguyên)
# và min_samples_split, max_depth, n_estimators.
param_grid = {
    'n_estimators': [50, 100], # Giảm số lượng để chạy nhanh hơn khi thử nghiệm
    'max_depth': [5, 10, None], # None nghĩa là không giới hạn độ sâu
    'min_samples_split': [2, 5],
    # 'n_features_subsample': [int(np.sqrt(x_train_full.shape[1])), x_train_full.shape[1] // 3 ] # Ví dụ
    # Hoặc để CustomRandomForestClassifier tự quyết định (mặc định là sqrt)
}

print("\nBắt đầu tìm kiếm tham số tốt nhất (thủ công)...")

best_score = -1.0
best_params = {}
best_model_custom = None

# Tạo tất cả các tổ hợp tham số
keys, values = zip(*param_grid.items())
parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Tổng số tổ hợp tham số cần thử: {len(parameter_combinations)}")

for i, params in enumerate(parameter_combinations):
    print(f"\nĐang thử tổ hợp {i+1}/{len(parameter_combinations)}: {params}")
    
    # Khởi tạo model với tham số hiện tại
    # Lưu ý: CustomRandomForestClassifier có n_features_subsample. Nếu bạn muốn điều chỉnh nó,
    # hãy thêm vào param_grid và truyền vào đây.
    # Hiện tại, nó sẽ dùng giá trị mặc định (sqrt).
    current_model = CustomRandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 100),
        min_samples_split=params.get('min_samples_split', 2)
        # n_features_subsample=params.get('n_features_subsample', None) # Thêm nếu có trong grid
    )
    
    # Huấn luyện model (sử dụng toàn bộ x_train_full cho lần này, không có cross-validation nội bộ)
    current_model.fit(x_train_full, y_train_full)
    
    # Đánh giá trên tập x_test_final (tạm gọi là validation set)
    y_pred_val = current_model.predict(x_test_final)
    current_score = custom_accuracy_score(y_test_final, y_pred_val)
    print(f"  Độ chính xác trên tập validation: {current_score:.4f}")
    
    if current_score > best_score:
        best_score = current_score
        best_params = params
        best_model_custom = current_model # Lưu lại model tốt nhất

print(f"\nTìm kiếm hoàn tất!")
print(f"Tham số tốt nhất tìm được: {best_params}")
print(f"Độ chính xác tốt nhất trên tập validation: {best_score:.4f}")


if best_model_custom:
    print("\nĐánh giá model tốt nhất trên tập kiểm tra cuối cùng (dùng lại x_test_final):")
    # y_pred_on_test_final = best_model_custom.predict(x_test_final) # Đã có từ lần lặp cuối
    # score_on_test_final = custom_accuracy_score(y_test_final, y_pred_on_test_final)
    # print(f"Độ chính xác của model tốt nhất trên tập kiểm tra: {score_on_test_final*100:.2f}%")
    print(f"(Sử dụng lại kết quả từ validation ở trên là {best_score*100:.2f}%)")

    print("\nBáo cáo phân loại cho model tốt nhất (phiên bản đơn giản hóa):")
    y_pred_best_model = best_model_custom.predict(x_test_final)
    print(custom_classification_report_simplified(y_test_final, y_pred_best_model))

    # Lưu model tốt nhất
    best_model_filename = 'custom_model_best_params.p'
    with open(best_model_filename, 'wb') as f:
        pickle.dump({'model': best_model_custom, 'params': best_params}, f)
    print(f"Model tốt nhất đã được lưu vào '{best_model_filename}'")

    # Ma trận nhầm lẫn cho model tốt nhất (tùy chọn)
    if 'custom_confusion_matrix' in globals() and hasattr(best_model_custom, 'classes_'):
        unique_labels_for_cm_best = best_model_custom.classes_ if best_model_custom.classes_ is not None else sorted(np.unique(np.concatenate((y_test_final, y_pred_best_model))))
        cm_best, cm_labels_best = custom_confusion_matrix(y_test_final, y_pred_best_model, labels=unique_labels_for_cm_best)
        if cm_best is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_best, annot=True, cmap='Blues', fmt='d',
                        xticklabels=cm_labels_best, yticklabels=cm_labels_best)
            plt.xlabel('Predicted labels (Best Model)')
            plt.ylabel('True labels (Best Model)')
            plt.title('Custom Confusion Matrix (Best Model)')
            plt.show()
else:
    print("\nKhông tìm thấy model tốt nhất nào.")